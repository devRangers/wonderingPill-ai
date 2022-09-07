import os
import numpy as np
import json
import cv2
import requests
from rembg import remove
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import keras

from labels.color_label import color_label
from labels.shape_label import shape_label

# Imports the Google Cloud client library
from google.cloud import vision
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './pill-text-recognition-a3e91afb350f.json'


from flask import Flask, jsonify, request

app = Flask(__name__)


def zoom_at(img, zoom, coord=None):
    h, w, _ = [ zoom * i for i in img.shape ]
    
    if coord is None: cx, cy = w/2, h/2
    else: cx, cy = [ zoom*c for c in coord ]
    
    img = cv2.resize( img, (0, 0), fx=zoom, fy=zoom)
    img = img[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
               int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)),
                : ] 
    return img

def crop_center(img):
    y, x, c = img.shape
    sx = x // 2-(min(x, y) // 2)
    sy = y // 2-(min(x, y) // 2)
    img = img[sy:sy+min(x,y), sx:sx+min(x,y)]

    return img

# 🔥 for url 
def preprocess_img(url):
    """ numpy -> rmbg -> zoom in"""
    # make image as a numpy array
    img_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
    input = cv2.imdecode(img_nparray, cv2.IMREAD_COLOR)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    # remove background
    rmbg_img = remove(input)[...,:3]
    rmbg_img = crop_center(rmbg_img)
    zoomed_img = zoom_at(rmbg_img, 1.5)

    return zoomed_img

def preprocess_img_for_model(img):
    img = cv2.resize(img, dsize=(224,224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return img

# 1️⃣ for path
def preprocess_img_path(path, output_path):
    # make image as a numpy array
    input = cv2.imread(path)
    # input = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    rmbg_img = remove(input)
    zoomed_img = zoom_at(rmbg_img, 1.5)
    resized_img = cv2.resize(zoomed_img, dsize=(224,224))
    
    cv2.imwrite(output_path, resized_img)


def get_letter(img):
    client = vision.ImageAnnotatorClient()
    # np array to bytes
    img = cv2.imencode('.jpg', img)[1].tobytes()

    img = vision.Image(content=img)

    response = client.text_detection(image=img)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    if texts:
        return texts[0].description
    return '-'


def load_model(model_path):
    loaded_model = keras.models.load_model(model_path)
    return loaded_model


# if __name__ == "__main__":
#     output_path = './testing_url_2.jpg'

#     input_url = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcT1z1x%2FbtrKktjAC6J%2FkSuZFm7K26Nc3gctzUzqDK%2Fimg.jpg"
#     img = preprocess_img(input_url)
#     cv2.imwrite(output_path, img) # 확인용
#     img_for_model = preprocess_img_for_model(img)

#     # shape classification
#     shape_classifier = load_model('./service/models/shape_classifier')
#     predict_result = shape_classifier.predict(img_for_model)
#     predict_shape_class = np.argmax(predict_result, axis=1)
#     print("Shape prediction: ", predict_shape_class)

#     # color classification
#     color_classifier = load_model('./service/models/color_classifier_2')
#     predict_result = color_classifier.predict(img_for_model)
#     predict_color_class = np.argmax(predict_result, axis=1)
#     predict_color_name = color_label[predict_color_class[0]]
#     print("Color prediction: ", predict_color_name)

#     # letter recognition
#     letter = get_letter(img)
#     letter = ''.join(letter.split())
#     print("letter: ", letter)


@app.route('/photos', methods=['POST'])
def inference():
    params = request.get_json()

    img_url = params["imgURL"]

    img = preprocess_img(img_url)
    img_for_model = preprocess_img_for_model(img)

    # shape classification
    shape_classifier = load_model('./service/models/shape_classifier_3_2')
    predict_result = shape_classifier.predict(img_for_model)
    predict_shape_class = np.argmax(predict_result, axis=1)
    predict_shape_name = shape_label[predict_shape_class[0]]

    # color classification
    color_classifier = load_model('./service/models/color_classifier_2')
    predict_result = color_classifier.predict(img_for_model)
    predict_color_class = np.argmax(predict_result, axis=1)
    predict_color_name = color_label[predict_color_class[0]]
    predict_color_names = []
    if '/' in predict_color_name:
        predict_color_names = predict_color_name.split('/')
    else:
        predict_color_names.append(predict_color_name)

    # letter recognition
    letter = get_letter(img)
    letter = ''.join(letter.split()).upper()

    result = {
        'shape' : predict_shape_name,
        'color' : predict_color_names,
        'letter' : letter
    }
    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)