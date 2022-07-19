from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

class MobileNet():

    def __init__(self, n_classes):
        self.input_shape = (224,224,3)
        self.n_classes = n_classes
        self.weights = 'imagenet'
        self.input_tensor = Input(shape = self.input_shape)
        self.base_model = MobileNetV2(include_top = False, weights = self.weights, input_tensor = self.input_tensor)

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        preds = Dense(units=self.n_classes, activation="softmax")(x)
        model = Model(inputs=self.input_tensor, outputs=preds)

        return model
        

        
