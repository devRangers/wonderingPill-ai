import pandas as pd
import os
import re
import pickle


def make_pill_df(image_dir):
    data = pd.read_csv("../data/pills_data.csv")
    paths, labels = [], []

    for dirname, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if '.jpg' in filename:
                file_path = dirname + "/" + filename
                paths.append(file_path)

                start_pos  = file_path.find("\\")
                end_pos = file_path.rfund("/")
                pill_code = file_path[start_pos+1:end_pos]

                pill_name = data[data["품목일련번호"] == int(pill_code)]["품목명"]
                pill_name = re.split('[/,-,(,).:]', str(pill_name))[0][5:].strip().replace('\nName','')
                labels.append(pill_name)
    
    data_df = pd.DataFrame({'path':paths, 'label':labels})
    return data_df


def make_pill_label(data_df):
    pill_dict = {}

    label_num = len(data_df['label_class'].value_counts())

    for i in range(label_num):
        pill_dict[i] = (data_df[data_df['label_class']==i]['label'].iloc[0])
    
    return pill_dict


if __name__ == "__main__":
    IMAGE_DIR = "../data/img"

    data_df = make_pill_df(IMAGE_DIR)
    data_df.to_csv("pills_data.preprocess.csv", index=False, encoding="utf-8")

    pill_dict = make_pill_df(data_df)
    with open("./label/pill_label.pkl", "wb") as tf:
        pickle.dump(pill_dict, tf)


