from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from val_model import get_val_model
from SceneClassifier.scene import scene_score
from moviepy.editor import VideoFileClip, concatenate_videoclips
from os.path import join
from Gen.va_gen import *
import matplotlib.pyplot as plt
import pickle
from heapq import *
from tensorflow.keras import backend as K
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("lowres")
parser.add_argument("highres")
parser.add_argument("out")
parser.add_argument("sec")
args = parser.parse_args()


thread = 0.8

total_score = 0
num = 0

def get_key(s):
    return s[0]
    

filename = args.lowres
hfilename = args.highres
with VideoFileClip(filename) as clip:
    duration = int(clip.duration)
print('Duration: {} sec.'.format(duration))

# load model
model = get_val_model('v1')
model.load_weights("{}-weights-{}.h5".format('VALModel','v1'), by_name=True)
model._make_predict_function()
# create score list
score = []
for i in range(0,duration-4,3):
    tname = 'tttmp/tmp.mp4'
    ffmpeg_extract_subclip(filename, i, i+4, targetname=tname)
    
    if scene_score(tname) < thread:
        score.append(0)
        print(0)
        continue
    else:
        x1batch_buf = [read_frames(tname)]
        np_x1 = preprocess_input(np.asarray(x1batch_buf))
        x2batch_buf = [read_audio(tname)]
        np_x2 = np.asarray(x2batch_buf)
        try:
            result = model.predict([np_x1,np_x2])
        except:
            del model
            K.clear_session()
            model = get_val_model('v1')
            model.load_weights("{}-weights-{}.h5".format('VALModel','v1'), by_name=True)
            model._make_predict_function()
            result = model.predict([np_x1,np_x2])
        num += 1
        total_score += result[0][0]
        print(result[0][0])
        score.append(result[0][0])
    
with open('score.pkl','wb') as fp:
        pickle.dump(score, fp)

blocks = []

s = None
maxi = 0

print("avg score = {}".format(total_score/num))

block_thres = float(input("block threshold: "))

for i in range(len(score)):
    if score[i] >= block_thres:
        if s is None:
            s = i
        else:
            maxi += score[i]
    else:
        if s is not None:
            st = s*3
            et = (i-1)*3 + 4
            duration = et - st
            maxi /= (i-s)
            heappush(blocks,(1-maxi,st,et,duration)) # because it's min-heap
            s = None
            maxi = 0

# clips selection

selecteds = []

time_remain = int(args.sec) # sec

while time_remain > 0:
    block = heappop(blocks)
    selected = (block[1],block[2],block[3],block[0])
    heappush(selecteds, selected)
    time_remain -= block[3]
    
selecteds.sort(key=get_key)

# make highlight
clips = []
for selected in selecteds:
    tname = 'tttmp\{}.mp4'.format(selected[0])
    ffmpeg_extract_subclip(hfilename, selected[0], selected[1], targetname=tname)
    clips.append(VideoFileClip(tname))


final_clip = concatenate_videoclips(clips)
final_clip.write_videofile(args.out)
