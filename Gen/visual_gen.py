import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from random import randint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

frame_shape = (320,180)

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
    xbatch_buf = []
    ybatch_buf = []
    while True:
        if len(xbatch_buf) == batch_size:
            np_x = preprocess_input(np.asarray(xbatch_buf))
            np_y = np.asarray(ybatch_buf)
            xbatch_buf = []
            ybatch_buf = []
            yield (np_x,np_y)
        cat = (cat+1)%2
        idx = idxs[cat]
        idxs[cat] += 1
        if idxs[cat] == len(files[cat]):
            idxs[cat] = 0
        if enreflex:
            reflex = randint(0,1)
        filename = join(paths[cat],files[cat][idx])
        xbatch_buf.append(read_frames(filename,reflex==1))
        ybatch_buf.append([cat])
        
    
    
    
if __name__ == '__main__':
    gen = data_gen()
    print(next(gen))