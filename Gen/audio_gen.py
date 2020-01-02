from moviepy.editor import *
import numpy as np
from librosa import feature
from os import listdir
from os.path import isfile, join
from random import randint

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
            np_x = np.asarray(xbatch_buf)
            #np_x = np.repeat(np_x[..., np.newaxis], 3, -1)
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
        xbatch_buf.append(read_audio(filename))
        ybatch_buf.append([cat])
    
    
if __name__ == '__main__':
    gen = data_gen()
    print(next(gen)[0].shape)
    