import pandas as pd
import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa

class Img_aug() :

    def __init__(self) :
        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.seq = iaa.Sequential(
            [               
                iaa.Fliplr(0.5), 
                iaa.Flipud(0.2), 

                self.sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                self.sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, 
                    rotate=(-45, 45), 
                    shear=(-16, 16),
                    order=[0, 1], 
                    cval=(0, 255),
                    mode=ia.ALL
                )),

                iaa.SomeOf((0, 5),
                    [
                        self.sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)),
                            iaa.AverageBlur(k=(2, 7)),
                            iaa.MedianBlur(k=(3, 11)), 
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), 
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), 
                      
                        iaa.SimplexNoiseAlpha(iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0.5, 1.0)),
                            iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                        ])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), 
                            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        iaa.Invert(0.05, per_channel=True),
                        iaa.Add((-10, 10), per_channel=0.5),
                        iaa.AddToHueAndSaturation((-20, 20)),
                        
                        iaa.OneOf([
                            iaa.Multiply((0.5, 1.5), per_channel=0.5),
                            iaa.FrequencyNoiseAlpha(
                                exponent=(-4, 0),
                                first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                second=iaa.ContrastNormalization((0.5, 2.0))
                            )
                        ]),
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), 
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                        self.sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                        self.sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), 
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

    data = pd.read_csv("./pills_data.csv")
    pill_code = pd.DataFrame(data['품목일련번호'])

    for i,row in pill_code.iterrows():
        # 1차적으로 100개
        if i == 100:
            break

        try:
            code = str(row['품목일련번호'])
            save_path = f'./img/{code}/'
            img = cv2.imread(f"./img/{code}/{code}_0.jpg")
            images_aug = aug.seq.augment_images([img for i in range(augment_num)])

            for num, aug_img in enumerate(images_aug):
                cv2.imwrite(f'{save_path}+{code}_{num+1}.jpg', aug_img)
        except:
            pass