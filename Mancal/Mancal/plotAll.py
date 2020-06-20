import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plot
import cv2
from tensorflow.keras.utils import plot_model
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn 

from model import get_Model_Sequential_Complete
import data
from data import getLabels, getTestLabels

### PLOT A RANDOM TRIAL FROM ALL CONDITIONS AVAILAVBLE ON THE DATA
def plotTrial():

    print('|' * 100)
    print('PLOT TRIAL')
    print('|' * 100)


    f,ax = plot.subplots(3,4) 

    f.suptitle("[increasing, decreasing, increasing then decreasing, decreasing then increasing]" , fontsize=8)
    
    plot.axis('tight')
    plot.subplots_adjust(wspace=0.8, hspace=0.8)
    
    nTrial = data.minorLabels.index('increasing')

    ax[0,0].plot(data.Healthy[nTrial])
    ax[0,0].set_title('Healthy')
    
    ax[1,0].plot(data.Inner[nTrial])
    ax[1,0].set_title('Inner')
    
    ax[2,0].plot(data.Outer[nTrial])
    ax[2,0].set_title('Outer')
    
    nTrial = data.minorLabels.index('decreasing')

    ax[0,1].plot(data.Healthy[nTrial])
    ax[0,1].set_title('Healthy')
    
    ax[1,1].plot(data.Inner[nTrial])
    ax[1,1].set_title('Inner')
    
    ax[2,1].plot(data.Outer[nTrial])
    ax[2,1].set_title('Outer')
    
    nTrial = data.minorLabels.index('increasing then decreasing')

    ax[0,2].plot(data.Healthy[nTrial])
    ax[0,2].set_title('Healthy')
    
    ax[1,2].plot(data.Inner[nTrial])
    ax[1,2].set_title('Inner')
    
    ax[2,2].plot(data.Outer[nTrial])
    ax[2,2].set_title('Outer')
    
    nTrial = data.minorLabels.index('decreasing then increasing')

    ax[0,3].plot(data.Healthy[nTrial])
    ax[0,3].set_title('Healthy')
    
    ax[1,3].plot(data.Inner[nTrial])
    ax[1,3].set_title('Inner')
    
    ax[2,3].plot(data.Outer[nTrial])
    ax[2,3].set_title('Outer')

    plot.xlim(0,data.nDataPoints)

    plot.savefig(data.modelDir + '/Trials.jpg', bbox_inches = 'tight')

    plot.close(f)


### PLOT A PICTURE OF THE LAYERS OF THE MDOEL
def plot_Model_Sequential_Complete():

    print('|' * 100)
    print('plot_Model_Sequential_Complete')
    print('|' * 100)

    model = get_Model_Sequential_Complete()

    plot_model(model,to_file=data.modelDir + '/modelo_completo.png',show_shapes=True)

### PLOT CONFUSION MATRIX
def plot_confusion_matrix():

    print('|' * 100)
    print('plot_confusion_matrix')
    print('|' * 100)

    class_names = ['Healthy', 'Inner', 'Outer']

    model = get_Model_Sequential_Complete()

    model.load_weights(data.modelDir + '/modelo_completo_pesos.hd5')

    Labels = getTestLabels()

    TS = np.concatenate((data.HealthyTest, data.InnerTest, data.OuterTest))

    if len(TS.shape) == 2:

        TS = np.expand_dims(TS, axis=0)
        TS = np.moveaxis(TS,0,-1)

    else:

        TS = np.moveaxis(TS,1,-1)

    test_prob = model.predict(TS)

    pred_labels = np.argmax(test_prob, axis = 1)

    CM = confusion_matrix(Labels, pred_labels)
    ax = plot.axes()
    sn.set(font_scale=1.4)
    sn.heatmap(CM, annot=False,annot_kws={"size": 16},  xticklabels=class_names, yticklabels=class_names, ax = ax)
    ax.set_title('Confusion matrix')
    plot.ylabel('True label')
    plot.xlabel('Predicted label')
    plot.savefig(data.modelDir + '/confusion_matrix.jpg', bbox_inches = 'tight')
    plot.close()