import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
import numpy as np
from keras.constraints import maxnorm
from keras.layers import Dense, Flatten
from keras.layers import Conv1D,MaxPooling2D, \
    MaxPooling1D, Embedding, Dropout,\
    GRU,TimeDistributed,Conv2D,\
    Activation,LSTM,Input
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
import tensorflow as tf
from keras import optimizers
import random



opt = 'adam'


def slice_batch(x, n_gpus, part):
    """
    Divide the input batch into [n_gpus] slices, and obtain slice number [part].
    i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
    """

    sh = K.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]

def to_multi_gpu(model, n_gpus=2):
    """
    Given a keras [model], return an equivalent model which parallelizes
    the computation over [n_gpus] GPUs.
    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor,
    hence the user sees a model that behaves the same as the original.
    """

    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name="input1")

    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch,
                             lambda shape: shape,
                             arguments={'n_gpus':n_gpus, 'part':g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = Concatenate(axis=0)(towers)

    return Model(inputs=[x], outputs=[merged])


def Build_Model_DNN_Image(shape, number_of_classes, layers_DNN, nodes_DNN, dropout):

    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense(nodes_DNN,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,layers_DNN-1):
        model.add(Dense(nodes_DNN,activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(number_of_classes, activation='softmax'))
    model_tmp = model

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model, model_tmp


def Build_Model_CNN_Image(shape, nclasses, layers_CNN, nodes_CNN, dropout):

    model = Sequential()

    model.add(Conv2D(nodes_CNN, (3, 3), padding='same', input_shape=shape))
    model.add(Activation('relu'))
    model.add(Conv2D(nodes_CNN, (3, 3)))
    model.add(Activation('relu'))

    for i in range(0,layers_CNN):
        model.add(Conv2D(nodes_CNN, (3, 3),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nclasses,activation='softmax',kernel_constraint=maxnorm(3)))
    model_tmp = model

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model, model_tmp



def Build_Model_RNN_Image(shape, nclasses, nodes_RNN, dropout):


    x = Input(shape=shape)

    # Encodes a row of pixels using TimeDistributed Wrapper.
    encoded_rows = TimeDistributed(LSTM(nodes_RNN,recurrent_dropout=dropout))(x)
    # Encodes columns of encoded rows.
    encoded_columns = LSTM(nodes_RNN, recurrent_dropout=dropout)(encoded_rows)

    # Final predictions and model.
    #prediction = Dense(256, activation='relu')(encoded_columns)
    prediction = Dense(nclasses, activation='softmax')(encoded_columns)
    model = Model(x, prediction)
    model_tmp = model

    model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
    return model,model_tmp

