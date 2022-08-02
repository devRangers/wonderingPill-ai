from dataloader.data_loader import get_train_valid_test, DataLoader
from tensorflow.keras.applications.xception import preprocess_input
from pill_classification.trainer import Trainer
import albumentations as A
from configs import mn_config


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
            "You need to specify what to train. (--shape_classifier or sth else)"
        )

    train_loader = DataLoader(
        X_train,
        y_train,
        image_size=IMAGE_SIZE,
        batch_size=config.batch_size,
        augmentor=None,  # use augementator later
        shuffle=True,
        pre_func=preprocess_input,
    )

    valid_loader = DataLoader(
        X_valid,
        y_valid,
        image_size=IMAGE_SIZE,
        batch_size=config.batch_size,
        augmentor=None,
        shuffle=False,
        pre_func=preprocess_input,
    )

    test_loader = DataLoader(
        X_test,
        y_test,
        image_size=IMAGE_SIZE,
        batch_size=config.batch_size,
        augmentor=None,
        shuffle=False,
        pre_func=preprocess_input,
    )

    train_batch = next(iter(train_loader))[0]
    valid_batch = next(iter(valid_loader))[0]
    test_batch = next(iter(test_loader))[0]

    print(train_batch.shape, valid_batch.shape, test_batch.shape)


if __name__ == "__main__":
    config = mn_config
    main(config)
