import tensorflow as tf
from . import slowfast
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model

__all__=['network']

def resnet50(inputs, **kwargs):
    model = slowfast.SlowFast_body(inputs, [3, 4, 6, 3], slowfast.bottleneck, **kwargs)
    return model

def resnet101(inputs, **kwargs):
    model = slowfast.SlowFast_body(inputs, [3, 4, 23, 3], slowfast.bottleneck, **kwargs)
    return model

def resnet152(inputs, **kwargs):
    model = slowfast.SlowFast_body(inputs, [3, 8, 36, 3], slowfast.bottleneck, **kwargs)
    return model

def resnet200(inputs, **kwargs):
    model = slowfast.Slow_body(inputs, [3, 24, 36, 3], slowfast.bottleneck, **kwargs)
    return model

def resnet30(inputs, **kwargs):
    model = slowfast.SlowFast_body(inputs, [3, 3, 6, 3], slowfast.bottleneck, **kwargs)
    return model

def get_model(out_dim):
    inputs = Input(shape=(24, 224, 224, 3))
    model = resnet50(inputs, num_classes=out_dim)	
    return model
    
def get_vis_model(depth='v1'):
    inputs = Input(shape=(24, 180, 320, 3))
    if depth == 'v1':
        model = resnet50(inputs, num_classes=2)	
    elif depth == 'v2':
        model = resnet101(inputs, num_classes=2)
    else:
        model = resnet152(inputs, num_classes=2)
    return model

network = {
    'resnet50':resnet50,
    'resnet101':resnet101,
    'resnet152':resnet152,
    'resnet200':resnet200
}

if __name__=="__main__":
    #tf.enable_eager_execution()
    x = tf.random_uniform([4, 64, 224, 224, 3])
    inputs = Input(shape=(24, 224, 224, 3))
    model = resnet50(inputs, num_classes=15)
    model.summary()
    plot_model(model, to_file='model.png')



