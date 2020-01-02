# coding: utf-8
import numpy as np
import cv2
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

def preprocess(frame):
    # 360p
    frame = cv2.resize(frame[:,180:540],(224,224), interpolation = cv2.INTER_NEAREST)
    frame = image.img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    return frame
