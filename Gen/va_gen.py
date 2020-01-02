import cv2
from moviepy.editor import *
import numpy as np
from librosa import feature
from os import listdir
from os.path import isfile, join
from random import randint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from math import floor
frame_shape = (320,180)



def read_audio(filename):
    '''
        input: video
        output: audio tensor
    '''
    reader = AudioFileClip(filename)
    audio = reader.to_soundarray()
    reader.close()
    
    # merge 2 channel
    audio = audio[:,0]/2 + audio[:,1]/2
    # get fixed size
    audio = audio[-176000:-1]

    mfcc = feature.mfcc(np.asfortranarray(audio),n_mfcc=40)
    #mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    mfcc = mfcc.reshape((40,344,1))
    return mfcc


def read_frames(filename,reflex=False):
    frame_buf = []
    video = cv2.VideoCapture(filename)
    framerate = float(video.get(cv2.CAP_PROP_FPS))
    skip_rate = floor(framerate/6 +0.01)
    ret, frame = video.read()
    count = 0
    while ret:
        if count % skip_rate == 0:
            resized = cv2.resize(frame, frame_shape, interpolation = cv2.INTER_AREA)
            if reflex:
                resized = cv2.flip(resized,1)
            frame_buf.append(resized)
        ret, frame = video.read()
        count += 1
    return frame_buf[0:24]
    
def data_gen(type='train',batch_size=1, enreflex=False):
    # read file list
    high_path = 'clips/highlights'
    non_path = 'clips/non-highlights'
    with open('clips/{}_n.csv'.format(type)) as f:
        nons = f.read().splitlines()
    
    with open('clips/{}_h.csv'.format(type)) as f:
        highs = f.read().splitlines()
    paths = [non_path,high_path]
    files = [nons,highs]
    idxs = [0,0]
    reflex = 0
    cat = 0
    # create buffer
    x1batch_buf = []
    x2batch_buf = []
    ybatch_buf = []
    while True:
        if len(x1batch_buf) == batch_size:
            np_x1 = preprocess_input(np.asarray(x1batch_buf))
            np_x2 = np.asarray(x2batch_buf)
            np_y = np.asarray(ybatch_buf)
            x1batch_buf = []
            x2batch_buf = []
            ybatch_buf = []
            yield ([np_x1,np_x2],np_y)
        cat = (cat+1)%2
        idx = idxs[cat]
        idxs[cat] += 1
        if idxs[cat] == len(files[cat]):
            idxs[cat] = 0
        if enreflex:
            reflex = randint(0,1)
        filename = join(paths[cat],files[cat][idx])
        x1batch_buf.append(read_frames(filename,reflex==1))
        x2batch_buf.append(read_audio(filename))
        ybatch_buf.append([cat])
    
    
    
if __name__ == '__main__':
    gen = data_gen()
    print(next(gen))