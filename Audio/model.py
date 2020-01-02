from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,GlobalAveragePooling2D,Dense,Dropout
from tensorflow import keras

def get_model():

    model = Sequential()
    model.add(Conv2D(filters=32,strides=(1,2), kernel_size=2, input_shape=(40,344,1), activation='relu',padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64,strides=(1,2), kernel_size=2, activation='relu',padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128,strides=(1,2), kernel_size=2, activation='relu',padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=128,strides=(2,2), kernel_size=2, activation='relu',padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=256,strides=(2,2), kernel_size=2, activation='relu',padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=512, kernel_size=2, activation='relu',padding='same'))
    model.add(Dropout(0.2))
    
    model.add(GlobalAveragePooling2D())

    model.add(Dense(1, activation='sigmoid'))
    return model
    
    
if __name__ == '__main__':
    model = get_model()
    model.summary()