'''


'''
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd
from multiprocessing import Pool

def convert_time(timestamp):
    times = timestamp.split(':')
    return int(times[0])*3600 + int(times[1])*60 + int(times[2])

def get_time_sequance(df):
    '''
        input: time interval should be in the first column of dataframe
        output: if there is no mistake of all time intervals
    '''
    total_time = 0
    timeseq = []
    for index, row in df.iterrows():
        timestamps = row['F'].split(',')
        t0 = convert_time(timestamps[0])
        t1 = convert_time(timestamps[1])
        total_time += t1-t0
        if t0 > t1:
            print(row['F'])
            return False
        timeseq.append((t0,t1))
    print(total_time)
    return timeseq

def get_counter_seq(seq):
    last_t = 0
    counter_seq = []
    for ts in seq:
        if ts[0]-last_t < 10:
            last_t = ts[1] + 1
            continue
        counter_seq.append((last_t,ts[0]-1))
        last_t = ts[1] + 1
    return counter_seq

def make_hclips(seq,vName, type, prefix,clip_size,overlap=1):
    count = 0
    for i in range(1,len(seq)-1):
        for t in range(seq[i][0],seq[i][1],clip_size-overlap):
            ffmpeg_extract_subclip(vName, t, t+clip_size, targetname="clips\{}\{}-{}.mp4".format(type,prefix,count))
            count += 1
        
def deal_full(i):
    name = 'V' + str(i)
    df = pd.read_csv('timestamps/{}.csv'.format(name))
    seq = get_time_sequance(df)
    if not seq:
        print(name, 'fails')
        return
    make_hclips(seq, 'fulls/{}.mp4'.format(name), 'highlights', name, 4)
    #counter_seq = get_counter_seq(seq)
    #make_hclips(counter_seq, 'fulls/{}.mp4'.format(name), 'non-highlights', name, 4)

if __name__ == '__main__':
    with Pool(4) as p:
        print(p.map(deal_full, [1, 2, 3]))
    
    
    