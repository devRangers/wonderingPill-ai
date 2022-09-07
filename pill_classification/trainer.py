import tensorflow as tf
from pill_classification.models.mobilenet import MobileNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

class Trainer():

    def __init__(self, config):
        self.config = config
        
        if self.config.gpu_id != -1:
            self.device = tf.test.gpu_device_name()
        else:
            self.device = "/CPU:0"
        
        self.model = MobileNet(n_classes= self.config.n_classes)

    def train(self, train_loader, valid_loader, X_train, X_valid):
        with tf.device(self.device):
            print("---학습 시작---")
            mobile_net = self.model
            model = mobile_net.forward()
            model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])

            rlr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, mode="min", verbose=self.config.verbose)
            ely_cb = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=self.config.verbose)

            history = model.fit(train_loader, epochs=self.config.n_epochs, steps_per_epoch=X_train.shape[0]//self.config.batch_size,
                                validation_data=valid_loader, validation_steps=X_valid.shape[0]//self.config.batch_size,
                                callbacks=([rlr_cb, ely_cb]), verbose=self.config.verbose)

        return model, history


        
