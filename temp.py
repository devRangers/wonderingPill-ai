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

    if config.shape_classifier is False:
        raise Exception(
            "You need to specify an what to train. (--shape_classifier or sth else)"
        )


if __name__ == "__main__":
    config = mn_config
    main(config)
