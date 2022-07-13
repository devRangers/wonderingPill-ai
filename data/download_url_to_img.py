import os
import urllib.request
import pandas as pd

def download_image(img_url, pill_name, save_path):
  result_url = './img/'
  os.mkdir(result_url + pill_name)
  urllib.request.urlretrieve(img_url, result_url + save_path + '.jpg')

if __name__ == "__main__":
    data = pd.read_csv("./pills_data.csv")
    image_url = pd.DataFrame(data['큰제품이미지'])

    for i,row in image_url.iterrows():
        if 'https://' in row['큰제품이미지']:
            # 1차적으로 100개 # FIX: 나중에 삭제
            if i == 100:
                break

            try:
                pill_name = str(data['품목일련번호'][i])
                save_path = f'{pill_name}/{pill_name}'

                download_image(row['큰제품이미지'], pill_name, save_path)
            except:
                pass