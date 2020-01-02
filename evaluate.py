import Audio.model
from Visual.nets import get_vis_model
from val_model import get_val_model
from va_model import get_va_model
from Gen.va_gen import *
from os.path import isfile, join
import numpy as np

from sklearn.metrics import accuracy_score,classification_report


high_path = 'clips/highlights'
non_path = 'clips/non-highlights'

def evaluate(label, pred):
    acc = accuracy_score(label, pred)
    target_names = ['Non-highlight', 'Highlight']
    print(classification_report(label, pred))
    return acc

def load_samples():
    # read file list
    with open('clips/all_n.csv'.format(type)) as f:
        nons = f.read().splitlines()
    with open('clips/train_n.csv'.format(type)) as f:
        tn = f.read().splitlines()
    with open('clips/val_n.csv'.format(type)) as f:
        vn = f.read().splitlines()
    nons = list(set(nons)-set(tn)-set(vn))
    with open('clips/test_h.csv'.format(type)) as f:
        highs = f.read().splitlines()
    
    return nons,highs

def predict_audio():
    count = 0
    nons, highs = load_samples()
    y_true = [0 for i in range(len(nons))] + [1 for i in range(len(highs))]
    model = Audio.model.get_model()
    model.load_weights("{}-weights.h5".format('Audio'), by_name=True)
    y_pred = []
    for i in range(len(nons)):
        count += 1
        print(count)
        filename = join(non_path,nons[i])
        xbatch_buf = [read_audio(filename)]
        np_x = np.asarray(xbatch_buf)
        result = model.predict(np_x)
        y_pred.append(int(result[0][0]>0.5))
    for i in range(len(highs)):
        count += 1
        print(count)
        filename = join(high_path,highs[i])
        xbatch_buf = [read_audio(filename)]
        np_x = np.asarray(xbatch_buf)
        result = model.predict(np_x)
        y_pred.append(int(result[0][0]>0.5))
    print(evaluate(y_true,y_pred))
    
def predict_visual():
    count = 0
    nons, highs = load_samples()
    y_true = [0 for i in range(len(nons))] + [1 for i in range(len(highs))]
    model = get_vis_model()
    model.load_weights("{}-weights-{}.h5".format('Visual','v1'), by_name=True)
    y_pred = []
    for i in range(len(nons)):
        count += 1
        print(count)
        filename = join(non_path,nons[i])
        xbatch_buf = [read_frames(filename)]
        np_x = preprocess_input(np.asarray(xbatch_buf))
        result = model.predict(np_x)
        y_pred.append(int(result[0][0]>0.5))
    for i in range(len(highs)):
        count += 1
        print(count)
        filename = join(high_path,highs[i])
        xbatch_buf = [read_frames(filename)]
        np_x = preprocess_input(np.asarray(xbatch_buf))
        result = model.predict(np_x)
        y_pred.append(int(result[0][0]>0.5))
    print(evaluate(y_true,y_pred))
    
def predict_va():
    count = 0
    nons, highs = load_samples()
    y_true = [0 for i in range(len(nons))] + [1 for i in range(len(highs))]
    model = get_va_model('v1')
    model.load_weights("{}-weights-{}.h5".format('VAModel','v1'), by_name=True)
    y_pred = []
    for i in range(len(nons)):
        count += 1
        print(count)
        filename = join(non_path,nons[i])
        x1batch_buf = [read_frames(filename)]
        np_x1 = preprocess_input(np.asarray(x1batch_buf))
        x2batch_buf = [read_audio(filename)]
        np_x2 = np.asarray(x2batch_buf)
        result = model.predict([np_x1,np_x2])
        y_pred.append(int(result[0][0]>0.5))
    for i in range(len(highs)):
        count += 1
        print(count)
        filename = join(high_path,highs[i])
        x1batch_buf = [read_frames(filename)]
        np_x1 = preprocess_input(np.asarray(x1batch_buf))
        x2batch_buf = [read_audio(filename)]
        np_x2 = np.asarray(x2batch_buf)
        result = model.predict([np_x1,np_x2])
        y_pred.append(int(result[0][0]>0.5))
    print(evaluate(y_true,y_pred))

def predict_val():
    count = 0
    nons, highs = load_samples()
    y_true = [0 for i in range(len(nons))] + [1 for i in range(len(highs))]
    model = get_val_model('v1')
    model.load_weights("{}-weights-{}.h5".format('VALModel','v1'), by_name=True)
    y_pred = []
    for i in range(len(nons)):
        count += 1
        print(count)
        filename = join(non_path,nons[i])
        x1batch_buf = [read_frames(filename)]
        np_x1 = preprocess_input(np.asarray(x1batch_buf))
        x2batch_buf = [read_audio(filename)]
        np_x2 = np.asarray(x2batch_buf)
        result = model.predict([np_x1,np_x2])
        y_pred.append(int(result[0][0]>0.5))
    for i in range(len(highs)):
        count += 1
        print(count)
        filename = join(high_path,highs[i])
        x1batch_buf = [read_frames(filename)]
        np_x1 = preprocess_input(np.asarray(x1batch_buf))
        x2batch_buf = [read_audio(filename)]
        np_x2 = np.asarray(x2batch_buf)
        result = model.predict([np_x1,np_x2])
        y_pred.append(int(result[0][0]>0.5))
    print(evaluate(y_true,y_pred))

if __name__ == '__main__':
    predict_audio()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    