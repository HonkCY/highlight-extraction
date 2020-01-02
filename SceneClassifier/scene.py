# coding: utf-8
import numpy as np
from .model import get_model
from .preprocess import preprocess
import cv2
from keras import backend as K

model = get_model()
# load weights
model.load_weights("SceneClassifier/weights.h5")
model._make_predict_function()


def scene_score(filename):
    global model
    global counter
    video = cv2.VideoCapture(filename)
    frame_buffer = []
    ret, frame = video.read()
    while ret:
        dealedframe = preprocess(frame)
        dealedframe = dealedframe.reshape((224,224,3))
        frame_buffer.append(dealedframe)
        ret, frame = video.read()
        
    np_frames = np.asarray(frame_buffer)
    try:
        scores = model.predict(np_frames)
    
    except:
        # to slove mem problem
        del model
        K.clear_session()
        model = get_model()
        # load weights
        model.load_weights("SceneClassifier/weights.h5")
        model._make_predict_function()
        scores = model.predict(np_frames)
    return np.average(scores.flatten())