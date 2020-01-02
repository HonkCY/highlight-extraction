import cv2
from moviepy.editor import *
import numpy as np
from librosa import feature
from os import listdir
from os.path import isfile, join
from random import randint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

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
    audio = (audio[:,0] + audio[:,1])/2
    
    # get fixed size
    audio = audio[-153501:-1]

    mfcc = feature.mfcc(np.asfortranarray(audio),n_mfcc=40)
    #mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    mfcc = mfcc.reshape((40,300,1))
    return mfcc


def read_frames(filename,reflex=False):
    frame_buf = []
    video = cv2.VideoCapture(filename)
    ret, frame = video.read()
    count = 0
    while ret:
        if count % 5 == 0:
            resized = cv2.resize(frame, frame_shape, interpolation = cv2.INTER_AREA)
            if reflex:
                resized = cv2.flip(resized,1)
            frame_buf.append(resized)
        ret, frame = video.read()
        count += 1
    return frame_buf
    
def data_gen(type='train',batch_size=1):
    # read file list
    high_path = 'clips/{}_highlights'.format(type)
    non_path = 'clips/{}_non-highlights'.format(type)
    nons = [f for f in listdir(non_path) if isfile(join(non_path, f))]
    highs = [f for f in listdir(high_path) if isfile(join(high_path, f))]
    paths = [non_path,high_path]
    files = [nons,highs]
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
        cat = randint(0,1)
        idx = randint(0,len(files[cat])-1)
        reflex = randint(0,1)
        filename = join(paths[cat],files[cat][idx])
        x1batch_buf.append(read_frames(filename,reflex==1))
        x2batch_buf.append(read_audio(filename))
        ybatch_buf.append([cat])
        
    
    
    
if __name__ == '__main__':
    gen = data_gen()
    print(next(gen))