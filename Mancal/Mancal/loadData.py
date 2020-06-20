import numpy as np
from mat4py import loadmat 
import random
from scipy.signal import resample
from os import listdir
from os.path import isfile, join

import data

### LOAD REAL DATASET
def load():

    print('|' * 100)
    print('LOAD DATA')
    print('|' * 100)

    onlyfiles = [f for f in listdir('datafiles') if isfile(join('datafiles', f))]

    for file in onlyfiles:
    
        print(file)

        f = loadmat('datafiles/' + file,'r')
        channel = f['Channel_1']
        flatList = [ item for elem in channel for item in elem]
        channel = np.array(flatList,dtype='Float32')

        channel = np.array(resample(channel,data.nDataPoints),dtype='Float32')

        char = file[0]
        run = file[4]

        
        if char == 'H':
        
            data.Healthy.append(channel)

        if char == 'I':
        
            data.Inner.append(channel)

        if char == 'O':

            data.Outer.append(channel)


    c = list(zip(data.Healthy, data.Inner, data.Outer, data.minorLabels))

    random.seed(data.Seed)
    random.shuffle(c)

    data.Healthy, data.Inner, data.Outer, data.minorLabels = zip(*c)

### GENERATE RANDOM DATASET, FOR DEBUG PURPOSE
def loadRandom():

    print('|' * 100)
    print('LOAD RANDOM DATA')
    print('|' * 100)

    onlyfiles = [f for f in listdir('datafiles') if isfile(join('datafiles', f))]

    for file in onlyfiles:
    
        print(file)

        channel = np.array(np.random.random((1,data.nDataPoints)))

        char = file[0]

        if char == 'H':
        
            data.Healthy.append(channel)

        if char == 'I':
        
            data.Inner.append(channel)

        if char == 'O':

            data.Outer.append(channel)