import tensorflow as tf
from . import slowfast_activity
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model

__all__=['network']

def resnet50(inputs, **kwargs):
    model = slowfast_activity.SlowFast_body(inputs, [3, 4, 6, 3], slowfast_activity.bottleneck, **kwargs)
    return model

def resnet101(inputs, **kwargs):
    model = slowfast_activity.SlowFast_body(inputs, [3, 4, 23, 3], slowfast_activity.bottleneck, **kwargs)
    return model

def resnet152(inputs, **kwargs):
    model = slowfast_activity.SlowFast_body(inputs, [3, 8, 36, 3], slowfast_activity.bottleneck, **kwargs)
    return model

def resnet200(inputs, **kwargs):
    model = slowfast_activity.Slow_body(inputs, [3, 24, 36, 3], slowfast_activity.bottleneck, **kwargs)
    return model

def resnet30(inputs, **kwargs):
    model = slowfast_activity.SlowFast_body(inputs, [3, 3, 6, 3], slowfast_activity.bottleneck, **kwargs)
    return model

def get_model(out_dim):
    inputs = Input(shape=(24, 224, 224, 3))
    model = resnet50(inputs, num_classes=out_dim)	
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



