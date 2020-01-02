from Audio.model import get_model
from Visual.nets import get_vis_model

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate,Dense

tune_layers = 0

def freezeLayer(layer):
    layer.trainable = False
    if hasattr(layer, 'layers'):
        for i in range(len(layer.layers)-tune_layers):
            freezeLayer(layer.layers[i])
            
def get_va_model(vdp='v2'):
    vinput = keras.Input(shape=(24, 180, 320, 3))
    ainput = keras.Input(shape=(40,344,1))
    
    vmodel = get_vis_model(vdp)
    amodel = get_model()
    
    vmodel.load_weights("{}-weights-{}.h5".format('Visual',vdp), by_name=True)
    amodel.load_weights("{}-weights.h5".format('Audio'), by_name=True)
    
    nvmodel = Model(vmodel.input,vmodel.layers[-2].output)
    namodel = Model(amodel.input,amodel.layers[-2].output)
    
    freezeLayer(nvmodel)
    freezeLayer(namodel)
    
    x = nvmodel(vinput)
    y = namodel(ainput)
    
    x = Dense(32,activation='relu')(x)
    y = Dense(32,activation='relu')(y)
    o = concatenate([x,y], axis=1)
    o = Dense(1,activation='sigmoid')(o)
    
    
    return Model([vinput,ainput],o)
    
    
    
    
    
if __name__ == '__main__':
    print(get_va_model().summary())