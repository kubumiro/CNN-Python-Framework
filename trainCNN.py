import numpy as np

import keras
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split
from keras.utils import np_utils

import time

import os
import glob
import h5py
import numpy as np
import hdf5tools
import models

n_classes=5


def GetData(path):
    print("Getting data from: {fn}".format(fn=path))

    filenames = glob.glob(path)
    pm = np.empty((0, 2, 80, 100, 1), dtype=np.uint8)
    lb = np.empty((0), dtype=np.uint8)

    for (i,fn) in enumerate(filenames):
        print("({ind}/{total}) Loading data from ... {f}".format(f=fn,ind = i+1,total = len(filenames)))
        cf = h5py.File(fn)
        pm = np.concatenate((pm, cf.get('pixelmaps')), axis=0)
        lb = np.concatenate((lb, cf.get('labels')))

    print("Pixelmaps shape: ",pm.shape)
    print("Labels shape: ",lb.shape)
    print("Number of samples: ",lb.shape[0])

    uq, hist = np.unique(lb, return_counts=True)
    print("Distribution of data: ",hist) 
    return pm,lb

def GetTrainingSet(path):

    pm, lb = GetData(path)
    number_of_classes = 5

    pmaps_train, pmaps_test, y_train, y_test = train_test_split(pm, lb, test_size=1 / 5, random_state=42)
    pmaps_valid, pmaps_test, y_valid, y_test = train_test_split(pmaps_test, y_test, test_size=1 / 2, random_state=42)

    X1_train0 = pmaps_train[:, 0]
    X2_train0 = pmaps_train[:, 1]
    X1_valid0 = pmaps_valid[:, 0]
    X2_valid0 = pmaps_valid[:, 1]
    X1_test0 = pmaps_test[:, 0]
    X2_test0 = pmaps_test[:, 1]

    X1_train = X1_train0.astype('float32') / 255
    X2_train = X2_train0.astype('float32') / 255
    X1_valid = X1_valid0.astype('float32') / 255
    X2_valid = X2_valid0.astype('float32') / 255
    X1_test = X1_test0.astype('float32') / 255
    X2_test = X2_test0.astype('float32') / 255

    y_train = np_utils.to_categorical(y_train, number_of_classes)
    y_valid = np_utils.to_categorical(y_valid, number_of_classes)
    y_test = np_utils.to_categorical(y_test, number_of_classes)

    return X1_train, X2_train, X1_valid, X2_valid, y_train, y_valid, X1_test, X2_test, y_test


def TrainCVN(model, path, name="CNN", predict=False):
    print("Training {fn}".format(fn=name))

    X1_train, X2_train, X1_valid, X2_valid, y_train, y_valid, X1_test, X2_test, y_test = GetTrainingSet(path)



    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc','top_k_categorical_accuracy'])
    from keras.callbacks import TensorBoard

    tb = TensorBoard(histogram_freq=0,
                     write_graph=True,
                     write_grads=False,
                     write_images=True)

    history = model.fit([X1_train,X2_train], y_train, batch_size=16, epochs=1, verbose=1,
                        validation_data=([X1_valid,X2_valid], y_valid), callbacks=[tb])
    print("Saving trained model {nm}...".format(nm=name))
    model.save('Models/model_{nm}.h5'.format(nm=name))

    if predict==True:
        if not os.path.exists("Predictions/{nm}/".format(nm=name)):
            os.makedirs("Predictions/{nm}/".format(nm=name))
        
        print("...Calculating predictions for traning set...")
        y_prob_train = model.predict([X1_train,X2_train])
        trainh5 = h5py.File("Predictions/{nm}/train".format(nm=name), 'w')
        trainh5.create_dataset('y_train_prob', data=y_prob_train)
        print("...Saving predictions for traning set to {pth}...".format(pth="Predictions/{nm}/train".format(nm=name)))

        print("...Calculating predictions for validation set...")
        y_prob_valid = model.predict([X1_valid,X2_valid])
        validh5 = h5py.File("Predictions/{nm}/valid.h5".format(nm=name), 'w')
        validh5.create_dataset('y_valid_prob', data=y_prob_valid)
        print("...Saving predictions for validation set to {pth}...".format(pth="Predictions/{nm}/valid".format(nm=name)))

            
        print("...Calculating predictions for testing set...")
        y_prob_test = model.predict([X1_test,X2_test])
        testh5 = h5py.File("Predictions/{nm}/test".format(nm=name), 'w')
        testh5.create_dataset('y_test_prob', data=y_prob_test)
        print("...Saving predictions for testing set to {pth}...".format(pth="Predictions/{nm}/test".format(nm=name)))

        
    else:
        print("...Done...")


