from SceneClassifier.scene import scene_score
from os import listdir,remove
from os.path import isfile, join

if __name__ == '__main__':
    path = 'clips/non-highlights'
    nons = [f for f in listdir(path) if isfile(join(path, f))]
    for filename in nons:
        try:
            score = scene_score(join(path,filename))
            print(filename,score)
            if  score < 0.8:
                remove(join(path,filename))
                print('del')
        except:
            remove(join(path,filename))
            print('del')