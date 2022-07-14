import numpy as np
import pandas as pd
import os
import re

IMAGE_DIR = "../data/img"

def make_pill_df(img_dir = IMAGE_DIR):
    data = pd.read_csv("../data/pills_data.csv")
    paths, labels = [], []

    for dirname, _, filenames in os.walk("../data/img"):
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