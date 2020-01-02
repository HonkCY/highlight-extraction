import pickle
from heapq import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips

filename = 'test720.mp4'

def get_key(s):
    return s[0]


with open('score.pkl','rb') as fp:
        score = pickle.load(fp)

blocks = []

s = None
maxi = 0

for i in range(len(score)):
        
    if score[i] >= 0.74:
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

time_remain = 180 # sec

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
    ffmpeg_extract_subclip(filename, selected[0], selected[1], targetname=tname)
    clips.append(VideoFileClip(tname))


final_clip = concatenate_videoclips(clips)
final_clip.write_videofile("my_concatenation.mp4")








