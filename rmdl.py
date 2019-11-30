import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import accuracy_score
import numpy as np
from keras.utils import np_utils

import gc
from sklearn.metrics import confusion_matrix
import collections
from sklearn.metrics import f1_score
import BuildModel as BuildModel
from trainCNN import GetTrainingSet
from keras.callbacks import ModelCheckpoint
np.random.seed(7)


def Image_Classification(path, shape, batch_size=128, epochs=[5, 5, 5],
                         layers_DNN = 5, nodes_DNN = 100,
                         nodes_RNN=2,
                         layers_CNN=5, nodes_CNN=16,
                         random_state=42, dropout=0.05):


    x_train, X2_train, x_valid, X2_valid, y_train, y_valid, x_test, X2_test, y_test = GetTrainingSet(path)
    #number_of_classes = np.shape(y_train)[1]
    number_of_classes = 5

    np.random.seed(random_state)

    y_proba = []
    score = []
    history_ = []


    model_DNN,model_tmp = BuildModel.Build_Model_DNN_Image(shape, number_of_classes, layers_DNN, nodes_DNN, dropout)
    filepath = "Models\RMDL\weights\weights_DNN.hdf5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    history = model_DNN.fit(x_train, y_train,
                            validation_data=(x_valid, y_valid),
                            epochs=epochs[0],
                            batch_size=batch_size,
                            callbacks=callbacks_list,
                            verbose=2)
    history_.append(history)
    model_tmp.load_weights(filepath)

    y_pr = model_tmp.predict_classes(x_test, batch_size=batch_size)
    y_proba.append(np.array(y_pr))
    model_tmp.save('Models/RMDL/model_DNN.h5')
    print(".........",y_test)
    print(".........", y_pr)

    del model_tmp
    del model_DNN
    gc.collect()



    model_RNN, model_tmp = BuildModel.Build_Model_RNN_Image(shape,
                                                            number_of_classes,
                                                            nodes_RNN,
                                                            dropout)

    filepath = "Models\RMDL\weights\weights_RNN.hdf5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    history = model_RNN.fit(x_train, y_train,
                            validation_data=(x_valid, y_valid),
                            epochs=epochs[1],
                            batch_size=batch_size,
                            verbose=2,
                            callbacks=callbacks_list)

    model_tmp.load_weights(filepath)
    history_.append(history)

    y_pr = model_tmp.predict(x_test, batch_size=batch_size)
    model_tmp.save('Models/RMDL/model_RNN.h5')
    y_pr = np.argmax(y_pr, axis=1)
    y_proba.append(np.array(y_pr))

    del model_tmp
    del model_RNN
    gc.collect()


    # reshape to be [samples][pixels][width][height]

    model_CNN, model_tmp = BuildModel.Build_Model_CNN_Image(shape,
                                                            number_of_classes,
                                                            layers_CNN,
                                                            nodes_CNN,
                                                            dropout)

    filepath = "weights\weights_CNN.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    history = model_CNN.fit(x_train, y_train,
                            validation_data=(x_valid, y_valid),
                            epochs=epochs[2],
                            batch_size=batch_size,
                            callbacks=callbacks_list,
                            verbose=2)
    history_.append(history)
    model_tmp.load_weights(filepath)
    y_pr = model_tmp.predict_classes(x_test, batch_size=batch_size)
    y_proba.append(np.array(y_pr))
    model_tmp.save('Models/RMDL/model_CNN.h5')
    del model_tmp
    del model_CNN
    gc.collect()

    y_proba = np.array(y_proba).transpose()
    print(y_proba.shape)
    final_y = []
    for i in range(0, y_proba.shape[0]):
        a = np.array(y_proba[i, :])
        a = collections.Counter(a).most_common()[0][0]
        final_y.append(a)

