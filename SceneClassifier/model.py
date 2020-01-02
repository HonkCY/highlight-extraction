# coding: utf-8
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.models import Sequential

def get_model():
    # return a scene classifier
    # build model
    vgg_model = VGG16(weights='imagenet', include_top=True) 

    model = Sequential()
    for layer in vgg_model.layers:
        model.add(layer)
    
    # remove prediction layer
    model.layers.pop()

    # add our prediction layer
    model.add(Dense(1, activation='relu'))
    
    # set loss function
    model.compile(loss='mean_squared_error',optimizer='adam')
    
    return model