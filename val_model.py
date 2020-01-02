import va_model
import Activity.nets_activity

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate,Dense,TimeDistributed
import tensorflow as ktf
from tensorflow.keras.layers import Lambda

tune_layers = 0

def freezeLayer(layer):
    layer.trainable = False
    if hasattr(layer, 'layers'):
        for i in range(len(layer.layers)-tune_layers):
            freezeLayer(layer.layers[i])
            
def get_val_model(vdp='v1'):
    vinput = keras.Input(shape=(24, 180, 320, 3))
    ainput = keras.Input(shape=(40,344,1))
    actinput = TimeDistributed(Lambda(lambda image: ktf.image.resize_images(image, (224, 224))))(vinput)
    
    vamodel = va_model.get_va_model(vdp)
    actmodel = Activity.nets_activity.get_model(11)
    
    vamodel.load_weights("{}-weights-{}.h5".format('VAModel',vdp), by_name=True)
    actmodel.load_weights("Activity.h5",by_name=True)
    
    nvamodel = Model(vamodel.input,vamodel.layers[-2].output)
    nactmodel = Model(actmodel.input,actmodel.layers[-2].output)
    
    freezeLayer(nvamodel)
    freezeLayer(nactmodel)
    
    x = nvamodel([vinput,ainput])
    y = nactmodel(actinput)
    
    x = Dense(32,activation='relu')(x)
    y = Dense(32,activation='relu')(y)
    o = concatenate([x,y], axis=1)
    o = Dense(1,activation='sigmoid')(o)
    
    
    return Model([vinput,ainput],o)
    
    
    
    
    
if __name__ == '__main__':
    print(get_val_model().summary())