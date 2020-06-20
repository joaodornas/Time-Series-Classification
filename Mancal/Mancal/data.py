
### MAIN GLOBAL VARIABLES USED ON THE PROGRAM

import numpy as np

nTraining = 0
totalTS = 12
nDataPoints = 0
Seed = 0
SeedTF = 0

modelDir = ''

minorLabels = ['increasing','increasing','increasing','decreasing','decreasing','decreasing','increasing then decreasing','increasing then decreasing','increasing then decreasing','decreasing then increasing','decreasing then increasing','decreasing then increasing']

Healthy = []
Inner = []
Outer = []

HealthyTrain = np.empty((nTraining,nDataPoints), dtype='Float32')
InnerTrain = np.empty((nTraining,nDataPoints), dtype='Float32')
OuterTrain = np.empty((nTraining,nDataPoints), dtype='Float32')

HealthyTest = np.empty((totalTS - nTraining,nDataPoints), dtype='Float32')
InnerTest = np.empty((totalTS - nTraining,nDataPoints), dtype='Float32')
OuterTest = np.empty((totalTS - nTraining,nDataPoints), dtype='Float32')

def getLabels(N):

    Labels = []

    for i in range(0,N):
        Labels.append(0)
        #'Healthy' --> 0

    for i in range(0,N):
        Labels.append(1)
        #'Inner' --> 1

    for i in range(0,N):
        Labels.append(2)
        #'Outer' --> 2

    Labels = np.array(Labels)

    return Labels

def getTestLabels():

    Labels = []

    for i in range(0,totalTS - nTraining):
        Labels.append(0)
        #'Healthy' --> 0

    for i in range(0,totalTS - nTraining):
        Labels.append(1)
        #'Inner' --> 1

    for i in range(0,totalTS - nTraining):
        Labels.append(2)
        #'Outer' --> 2

    Labels = np.array(Labels)

    return Labels