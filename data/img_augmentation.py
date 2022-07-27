import pandas as pd
import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa

class Img_aug :

    def __init__(self) :
        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.seq = iaa.Sequential(
            [   
                iaa.Fliplr(0.5),

                self.sometimes(iaa.Resize((0.5, 1.0))),
            
                self.sometimes(iaa.Affine(
                    scale=(0.5, 0.9),
                    rotate=(-45, 45),
                )),


                iaa.SomeOf((0, 5),
                    [
                        iaa.OneOf([
                            iaa.BilateralBlur(d=(3,10), sigma_color=(10,250), sigma_space=(10,250))
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.2)), 
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.0)), 
                    
                        iaa.SimplexNoiseAlpha(iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0.5, 1.0)),
                            iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                        ])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

                        iaa.AddToHueAndSaturation((-20, 20)),
                        
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                        self.sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.0), sigma=0.1)),
                        self.sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

if __name__ == "__main__":
    aug = Img_aug()		
    augment_num = 100 # augmentation 이미지의 갯수

    data = pd.read_csv("./pills_data.available_in_api.csv")
    pill_code = pd.DataFrame(data['품목일련번호'])

    for i,row in pill_code.iterrows():
        try:
            code = str(row['품목일련번호'])
            save_path = f'./img/{code}/'
            img = cv2.imread(f"./img/{code}/{code}_0.jpg")
            images_aug = aug.seq.augment_images([img for i in range(augment_num)])

            for num, aug_img in enumerate(images_aug):
                cv2.imwrite(f'{save_path}{code}_{num+1}.jpg', aug_img)
        except:
            pass