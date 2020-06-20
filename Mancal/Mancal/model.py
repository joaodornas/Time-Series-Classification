
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plot
from os import path
import random
import math

import tensorflow.keras.models as Models
import tensorflow.keras.layers as Layers
import tensorflow.keras.optimizers as Optimizer
import tensorflow as tf

import data
from data import getLabels, getTestLabels

### PREPROCESS THE DATA
### THE TRAINING DATA AND THE TEST DATA ARE Z-SCORED USING THE MEAN AND STD FROM TRAINING DATA ONLY
def preprocessData():

    print('|' * 100)
    print('preprocessData')
    print('|' * 100)

    data.HealthyTrain = np.array(data.Healthy[:data.nTraining])
    data.InnerTrain = np.array(data.Inner[:data.nTraining])
    data.OuterTrain = np.array(data.Outer[:data.nTraining])

    data.HealthyTest = np.array(data.Healthy[data.nTraining:])
    data.InnerTest = np.array(data.Inner[data.nTraining:])
    data.OuterTest = np.array(data.Outer[data.nTraining:])

    healthy_mean = np.mean(data.HealthyTrain)
    inner_mean = np.mean(data.InnerTrain)
    outer_mean = np.mean(data.OuterTrain)

    healthy_std = np.std(data.HealthyTrain)
    inner_std = np.std(data.InnerTrain)
    outer_std = np.std(data.OuterTrain)

    data.HealthyTrain = (data.HealthyTrain - healthy_mean) / healthy_std
    data.InnerTrain = (data.InnerTrain - inner_mean) / inner_std
    data.OuterTrain = (data.OuterTrain - outer_mean) / outer_std

    data.HealthyTest = (data.HealthyTest - healthy_mean) / healthy_std
    data.InnerTest = (data.InnerTest - inner_mean) / inner_std
    data.OuterTest = (data.OuterTest - outer_mean) / outer_std

### MODEL DEFINITION
def get_Model_Sequential_Complete():
 
    #print('|' * 100)
    #print('get_Model_Sequential_Complete')
    #print('|' * 100)

    tf.random.set_seed(data.SeedTF)

    model = Models.Sequential()

    model.add(Layers.Conv1D(128,kernel_size=(8),input_shape=(data.nDataPoints,1)))
    model.add(Layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Layers.ReLU())

    model.add(Layers.Conv1D(256,kernel_size=(5),input_shape=(data.nDataPoints,1)))
    model.add(Layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Layers.ReLU())

    model.add(Layers.Conv1D(128,kernel_size=(3),input_shape=(data.nDataPoints,1)))
    model.add(Layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Layers.ReLU())

    model.add(Layers.GlobalMaxPool1D(data_format='channels_last'))
    model.add(Layers.Softmax(axis=-1))

    model.compile(optimizer=Optimizer.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model

### TRAIN THE MODEL USING THE DATA
def fit_Model_Sequential_Complete():

    print('|' * 100)
    print('fit_Model_Sequential_Complete')
    print('|' * 100)

    nTrain = data.nTraining - math.ceil(data.nTraining * 0.30)

    ts_healthy_train = data.HealthyTrain[:nTrain,:]
    ts_healthy_valid = data.HealthyTrain[nTrain:,:]

    ts_inner_train = data.InnerTrain[:nTrain,:]
    ts_inner_valid = data.InnerTrain[nTrain:,:]

    ts_outer_train = data.OuterTrain[:nTrain,:]
    ts_outer_valid = data.OuterTrain[nTrain:,:]

    TS_train = np.concatenate((ts_healthy_train, ts_inner_train, ts_outer_train), axis=0)
    TS_valid = np.concatenate((ts_healthy_valid, ts_inner_valid, ts_outer_valid), axis=0)

    if len(TS_train.shape) == 2:

        TS_train = np.expand_dims(TS_train, axis=0)
        TS_train = np.moveaxis(TS_train,0,-1)

        TS_valid = np.expand_dims(TS_valid, axis=0)
        TS_valid = np.moveaxis(TS_valid,0,-1)

    else:

        TS_train = np.moveaxis(TS_train,1,-1)
        TS_valid = np.moveaxis(TS_valid,1,-1)

    Labels_train = getLabels(nTrain)
    Labels_valid = getLabels(data.nTraining - nTrain)

    c = list(zip(TS_train, Labels_train))

    random.seed(data.Seed)
    random.shuffle(c)

    TS_train, Labels_train = zip(*c)

    c = list(zip(TS_valid, Labels_valid))

    random.seed(data.Seed)
    random.shuffle(c)

    TS_valid, Labels_valid = zip(*c)

    model = get_Model_Sequential_Complete()

    model.summary()

    TS_train = np.array(TS_train)
    Labels_train = np.array(Labels_train)
    TS_valid = np.array(TS_valid)
    Labels_valid = np.array(Labels_valid)

    tf.random.set_seed(data.SeedTF)

    trained = model.fit(TS_train,Labels_train,batch_size=data.nTraining,epochs=10,validation_data=[TS_valid, Labels_valid])

    if not (path.exists(data.modelDir + '/modelo_completo.hd5')):
        os.mkdir(data.modelDir + '/modelo_completo.hd5')

    model.save(data.modelDir + '/modelo_completo.hd5')
    model.save_weights(data.modelDir + '/modelo_completo_pesos.hd5')

    plot.plot(trained.history['accuracy'])
    plot.plot(trained.history['val_accuracy'])
    plot.title('Model Accuracy')
    plot.ylabel('Accuracy')
    plot.xlabel('Epoch')
    plot.legend(['Train', 'Validation'], loc='upper left')
    plot.savefig(data.modelDir + '/Model_Accuracy.jpg', bbox_inches = 'tight')
    plot.close()

    plot.plot(trained.history['loss'])
    plot.plot(trained.history['val_loss'])
    plot.title('Model Loss')
    plot.ylabel('Loss')
    plot.xlabel('Epoch')
    plot.legend(['Train', 'Validation'], loc='upper left')
    plot.savefig(data.modelDir + '/Model_Loss.jpg', bbox_inches = 'tight')
    plot.close()

### EVALUATE MODEL PERFORMANCE
def evaluate_Model_Sequential_Complete():

    print('|' * 100)
    print('evaluate_Model_Sequential_Complete')
    print('|' * 100)

    Labels = getTestLabels()

    TS = np.concatenate((data.HealthyTest, data.InnerTest, data.OuterTest), axis=0)

    if len(TS.shape) == 2:

        TS = np.expand_dims(TS, axis=0)
        TS = np.moveaxis(TS,0,-1)

    else:

        TS = np.moveaxis(TS,1,-1)

    model = get_Model_Sequential_Complete()

    model.load_weights(data.modelDir + '/modelo_completo_pesos.hd5')

    model.evaluate(TS,Labels, verbose=1)