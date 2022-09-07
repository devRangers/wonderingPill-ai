import os
import urllib.request
import pandas as pd

def download_image(img_url, pill_name, save_path):
    """
    Args:
        img_url : 다운로드 url 경로(부모 폴더)
        pill_code : 다운로드할 알약 code번호
        save_path : 다운로드 이미지를 저장할 경로 뒷부분(자식폴더)
    """
    result_url = './img/'
    os.mkdir(result_url + pill_name)
    urllib.request.urlretrieve(img_url, result_url + save_path + '.jpg')

if __name__ == "__main__":
    data = pd.read_csv("./pills_data.available_in_api.csv")
    image_url = pd.DataFrame(data['큰제품이미지'])

    for i,row in image_url.iterrows():
        if 'https://' in row['큰제품이미지']:

            try:
                pill_name = str(data['품목일련번호'][i])
                save_path = f'{pill_name}/{pill_name}'

                download_image(row['큰제품이미지'], pill_name, save_path)
            except:
                pass