# coding: utf-8
import numpy as np
import cv2
from .preprocess import *
from .model import *
import matplotlib.pyplot as plt

# to fit autoencoder
frames_base = 16
snippet_size = 32

def get_ingame_array(videopath, batch_size=1,plotting=False):
    batch_size = snippet_size//frames_base * batch_size
    model = get_model()
    frames_buffer = []
    val_buffer = []
    vals = []
    video = cv2.VideoCapture(videopath)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(length):
        _, frame = video.read()
        if frame is None:
            break
        print("{}/{}".format(i+1,length))
        frames_buffer.append(preprocess(frame))
        if len(frames_buffer) == frames_base:
            np_frames = np.asarray(frames_buffer).reshape((frames_base,224,224,3))
            results = model.predict(np_frames)
            val_buffer.append(np.mean(results))
            frames_buffer = []
        if len(val_buffer) == batch_size:
            np_vals = np.asarray(val_buffer)
            vals.append(np.mean(np_vals))
            val_buffer = []

    if plotting:
        plt.plot(vals)
        plt.title('Video values')
        plt.ylabel('in game rate')
        plt.xlabel('snippets(frames={}x{})'.format(32,batch_size))
        plt.legend(['in game rate'], loc='upper left')
        plt.show()
    return vals