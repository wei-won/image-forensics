# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
from __future__ import print_function
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import keras
import tensorflow as tf
import numpy as np
from keras.utils.io_utils import HDF5Matrix
from sklearn.metrics import precision_recall_fscore_support
from FullyConvNetCore import load_FCN_model
import keras.backend as K


'''
###
Changes here:
ref: https://github.com/dmlc/xgboost/issues/1715
Added the line below to avoid 'KMP duplicate lib' error. (Seems to be Mac specific problem)
Alternative solution: install 'MKL Optimizations' packages with 'conda install nomkl'.
###
'''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 64
epochs = 400


def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy


FCN = load_FCN_model('pretrained_busterNet.hd5')
optimizer = keras.optimizers.Adam(lr=0.01, )
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=10, min_lr=0.001)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                           restore_best_weights=True)
# FCN.compile(optimizer='adam', loss='binary_crossentropy')
FCN.compile(optimizer='adam', loss=create_weighted_binary_crossentropy(1,10), metrics=['accuracy'])

X = HDF5Matrix('../../data/CASIA/CASIA-CMFD-Pos.hd5', 'X')
Y = HDF5Matrix('../../data/CASIA/CASIA-CMFD-Pos.hd5', 'Y')


X = X.data + np.array([103.939, 116.779, 123.68]).reshape([1, 1, 1, 3])
X = X/255

X_train = X[:1000, :, :, :]
X_test = X[1000:, :, :, :]
Y_train = Y.data[:1000, :, :, 0][:, :, :, np.newaxis]
Y_test = Y.data[1000:, :, :, 0][:, :, :, np.newaxis]

FCN.fit(X_train, Y_train,
           batch_size=batch_size,
           epochs=epochs,
           verbose=1,
           callbacks=[reduce_lr, early_stop])

FCN.save('fcn_model.hd5')
