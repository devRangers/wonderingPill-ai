import numpy as np
import pandas as pd
import os
import re
import pickle


def make_pill_df(image_dir):
    data = pd.read_csv("../data/pills_data.available_in_api.csv")
    paths, codes, names, shapes, colors, fronts, backs = [], [], [], [], [], [], []

    for dirname, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if '.jpg' in filename:
                file_path = dirname + "/" + filename
                paths.append(file_path)

                start_pos  = file_path.find("\\")
                end_pos = file_path.rfind("/")
                pill_code = file_path[start_pos+1:end_pos]
                codes.append(pill_code)

                pill_name = data[data["품목일련번호"] == int(pill_code)]["품목명"]
                pill_name = re.split('[/,-,(,).:]', str(pill_name))[0][5:].strip().replace('\nName','')
                names.append(pill_name)

                shape = data[data["품목일련번호"] == int(pill_code)]["의약품제형"].tolist()[0]
                shapes.append(shape)

                color = data[data["품목일련번호"] == int(pill_code)]["색상"].tolist()[0]
                colors.append(color)

                front = data[data["품목일련번호"] == int(pill_code)]["표시앞"].tolist()[0]
                fronts.append(front)
                back = data[data["품목일련번호"] == int(pill_code)]["표시뒤"].tolist()[0]
                backs.append(back)
    
    data_df = pd.DataFrame({'path':paths, 'code':codes, 'name':names, 'shape':shapes, 'color':colors, 'front':fronts, 'back':backs})

    data_label = pd.get_dummies(data_df['shape']).values
    data_df['shape_class'] = np.argmax(data_label, axis=1)

    data_label = pd.get_dummies(data_df['color']).values
    data_df['color_class'] = np.argmax(data_label, axis=1)

    data_label = pd.get_dummies(data_df['front']).values
    data_df['front_class'] = np.argmax(data_label, axis=1)

    data_label = pd.get_dummies(data_df['back']).values
    data_df['back_class'] = np.argmax(data_label, axis=1)
    return data_df


def make_pill_label(data_df):
    pill_dict = {}

    label_num = len(data_df['shape_class'].value_counts())

    for i in range(label_num):
        pill_dict[i] = (data_df[data_df['shape_class']==i]['shape'].iloc[0])

    return pill_dict


if __name__ == "__main__":
    IMAGE_DIR = "../data/img"

    data_df = make_pill_df(IMAGE_DIR)
    data_df.to_csv("pills_data.available_in_api.preprocess.csv", index=False, encoding="euc-kr")

    pill_dict = make_pill_label(data_df)
    with open("./label/pill_label.pkl", "wb") as tf:
        pickle.dump(pill_dict, tf)


