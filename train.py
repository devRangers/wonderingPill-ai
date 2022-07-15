from pill_classification.data_loader import get_train_valid_test, DataLoader
from tensorflow.keras.applications.xception import preprocess_input
import albumentations as A
from configs import mn_config

DF_PATH = "./data/pills_data.preprocess.csv"
IMAGE_SIZE = 224

def main(config):
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_train_valid_test(df_path=DF_PATH, valid_size=0.1)

    augmentor_Flip = A.Compose([
         A.HorizontalFlip(p=0.5)
    ])

    train_loader = DataLoader(X_train, y_train,
                                image_size=IMAGE_SIZE,
                                batch_size=config.batch_size,
                                augmentor=augmentor_Flip,
                                shuffle=True,
                                pre_func = preprocess_input)

    valid_loader = DataLoader(X_valid, y_valid,
                                image_size=IMAGE_SIZE,
                                batch_size=config.batch_size,
                                augmentor=None,
                                shuffle=False,
                                pre_func=preprocess_input)

    test_loader = DataLoader(X_test, y_test,
                            image_size=IMAGE_SIZE,
                            batch_size=config.batch_size,
                            augmentor=None,
                            shuffle=False,
                            pre_func=preprocess_input)
    
    train_batch = next(iter(train_loader))[0]
    valid_batch = next(iter(valid_loader))[0]
    test_batch = next(iter(test_loader))[0]

    print(train_batch.shape, valid_batch.shape, test_batch.shape)

if __name__ == "__main__":
    config = mn_config
    main(config)