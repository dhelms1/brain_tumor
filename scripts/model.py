from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
IMG_SIZE = 224

def EfficientNetClassifier():
    '''
    Using transfer learning, create an EfficientNetB0 model to be used. We
    will update the output for the model as well as unfreeze some of the final
    15 layers to be trainined and learn more low level features in our images.
    '''
    effnetb0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model = layers.GlobalAveragePooling2D()(effnetb0.output)
    model = layers.Dropout(0.3)(model)
    model = layers.Dense(4, activation='softmax')(model)
    model = models.Model(inputs=effnetb0.input, outputs=model)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    