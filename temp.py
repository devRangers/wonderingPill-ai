from dataloader.data_loader import get_train_valid_test, DataLoader
from tensorflow.keras.applications.xception import preprocess_input
from pill_classification.trainer import Trainer
import albumentations as A
from configs import mn_config

DF_PATH = "./data/pills_data.preprocess.csv"
IMAGE_SIZE = 224


def main(config):
    if config.shape_classifier == True:
        DF_PATH = "./data/pills_data.shape.balanced.csv"
        N_CLASSES = 9

        X_train, X_valid, X_test, y_train, y_valid, y_test = get_train_valid_test(
            df_path=DF_PATH, valid_size=0.1
        )
        print(X_train.shape, X_valid.shape, X_test.shape)
        print(y_train.shape, y_valid.shape, y_test.shape)


if __name__ == "__main__":
    config = mn_config
    main(config)
