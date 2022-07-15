import sklearn
import cv2
import numpy as np
import pandas as pd
import tensorflow
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence


def get_train_valid_test(df_path, valid_size=0.1, random_state=216):
    data_df = pd.read_csv(df_path, encoding="utf-8")
    data_df['path'] = data_df['path'].replace('\.+', '.', regex=True) 

    data_path = data_df['path'].values
    data_label = pd.get_dummies(data_df['label']).values

    X_train, X_test, y_train, y_test = train_test_split(data_path, data_label, test_size=valid_size, stratify=data_label, random_state=random_state)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=(valid_size / (1-valid_size)), stratify=y_train, random_state=random_state)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


class DataLoader(Sequence):

    def __init__(self, image_filenames, labels, image_size=224, batch_size=64, augmentor=None, shuffle=False, pre_func=None):
        """
        :params image_filenames: image 경로들
        :params labels: image의 라벨들
        :shuffle: epoch 종료 시마다 데이터를 섞을지 여부(train data)
        """
        self.image_filenames = image_filenames
        self.labels = labels
        self.image_size = image_size
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.pre_func = pre_func
        self.shuffle = shuffle

        if self.shuffle:
            pass

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))
    
    def __getitem__(self, index):
        image_name_batch = self.image_filenames[index*self.batch_size:(index+1)*self.batch_size]
        if self.labels is not None:
            label_batch = self.labels[index*self.batch_size:(index+1)*self.batch_size]

        image_batch = np.zeros((image_name_batch.shape[0], self.image_size, self.image_size, 3), dtype="float32")

        for image_index in range(image_name_batch.shape[0]):
            image = cv2.cvtColor(cv2.imread(image_name_batch[image_index]), cv2.COLOR_BGR2RGB)

        if self.augmentor is not None:
            image = self.augmentor(image=image)['image']

        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.pre_func is not None:
            image = self.pre_func(image)
        
        image_batch[image_index] = image
        return image_batch, label_batch
    
    def on_epoch_end(self):
        if(self.shuffle):
            self.image_filenames, self.labels = sklearn.utils.shuffle(self.image_filenames, self.labels)
        else:
            pass