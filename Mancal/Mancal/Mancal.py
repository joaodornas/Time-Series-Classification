
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os import path
import cv2
import numpy as np
import matplotlib.pyplot as plot

import data
from loadData import load, loadRandom
from plotAll import plotTrial, plot_Model_Sequential_Complete, plot_confusion_matrix
from model import preprocessData, fit_Model_Sequential_Complete, evaluate_Model_Sequential_Complete

### THIS FUNCTION RUNS THE COMPLETE MODEL (LOADING DATA, PREPROCESSING, FITTING MODEL, EVALUATE)
def run_Complete_Training_Model(modelDir,dataPoints,nTrain):

    data.modelDir = modelDir

    if not (path.exists(data.modelDir)):
        os.mkdir(data.modelDir)

    ### DEFINE THE AMOUNT OF DATA POINTS AND SEED NUMBER AND HOW MANY TRAINING TIME SERIES
    data.nDataPoints = int(dataPoints)
    data.Seed = 6
    data.SeedTF = 6
    data.nTraining = int(nTrain)

    ### LOAD ALL DATA INTO MEMORY
    load()
    # loadRandom() # THIS FUNCTION LOADS RANDOM DATA SET, FOR DEBUGGING PURPOSE

    ### PLOT RANDOM TRIAL
    plotTrial()

    ### TRAIN AND PLOT MODEL
    preprocessData()

    ### GENERATE A FIGURE WITH A DRAW OF THE MODEL
    plot_Model_Sequential_Complete()

    ### FIT THE MODEL (TRAIN) USING THE DATA
    fit_Model_Sequential_Complete()

    ### EVALUATE MODEL PERFORMANCE
    evaluate_Model_Sequential_Complete()

    ### PLOT CONFUSION MATRIX
    plot_confusion_matrix()

    ### THIS FUNCTION SHOWS ALL FIGURES FROM RESULT ON THE SCREEN
def run_Show_Results(dir):

    data.modelDir = dir

    model_image = cv2.imread(data.modelDir + '/modelo_completo.png')
    model_image = cv2.resize(model_image, (600, 1200))

    trial_image = cv2.imread(data.modelDir + '/Trials.jpg')
    trial_image = cv2.resize(trial_image, (600, 600))

    accuracy_image = cv2.imread(data.modelDir + '/Model_Accuracy.jpg')
    accuracy_image = cv2.resize(accuracy_image, (600, 600))

    loss_image = cv2.imread(data.modelDir + '/Model_Loss.jpg')
    loss_image = cv2.resize(loss_image, (600, 600))

    confusion_image = cv2.imread(data.modelDir + '/Confusion_Matrix.jpg')
    confusion_image = cv2.resize(confusion_image, (600, 600))

    vis1 = np.concatenate((trial_image, confusion_image), axis=0)
    vis2 = np.concatenate((accuracy_image, loss_image), axis=0)
    vis3 = np.concatenate((vis1, vis2), axis=1)
    vis4 = np.concatenate((model_image, vis3), axis=1)

    results = np.array(vis4)

    f,ax = plot.subplots(1,1) 
    ax.imshow(results)
    ax.set_title('Results')
    ax.axis('off')
    plot.show()




