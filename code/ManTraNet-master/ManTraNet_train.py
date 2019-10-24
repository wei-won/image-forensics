import os
import sys
import numpy as np
import keras
import keras.backend as K
from keras.utils.io_utils import HDF5Matrix

# import cv2
# import requests
# from PIL import Image
# from io import BytesIO
# from datetime import datetime
# from matplotlib import pyplot


manTraNet_root = './'
manTraNet_srcDir = os.path.join(manTraNet_root, 'src')
sys.path.insert(0, manTraNet_srcDir)
import modelCore
manTraNet_modelDir = os.path.join(manTraNet_root, 'pretrained_weights')
checkpoint_dir = 'ckpt'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
ckpt_mdl = os.path.join(checkpoint_dir, 'model_10.hd5')     # 'model_10.hd5'


'''
Fine tune configuration: 
Freeze part of the network when training.
Determine loss function.
- freeze_featex: Freeze feature extractor (1st half) and train the location net (2nd half)
- freeze_locNet: Freeze location net (2nd half) and train the feature extractor (1st half)
- wtd_bi_ce: Apply weighted_binary_crossentropy when True; Apply binary_crossentropy when False.
- loss_wt: Determines [zero_weight, one_weight] when using weighted_binary_crossentropy.
- batch_size: Batch size for training epochs.
- epochs: Maximum number of epochs for training.
- lr: Learning rate.
'''
freeze_featex = True
freeze_locNet = False
wtd_bi_ce = False
loss_wt = [0.1, 1]
batch_size = 4
epochs = 50
lr = 0.0001


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


'''
Load A Pre-trained ManTraNet Model
'''

manTraNet = modelCore.load_pretrain_model_by_index( 4, manTraNet_modelDir )
manTraNet.load_weights(ckpt_mdl)

if freeze_featex:
    manTraNet.get_layer('Featex').trainable = False

if freeze_locNet:
    manTraNet.get_layer('outlierTrans').trainable = False
    manTraNet.get_layer('bnorm').trainable = False
    manTraNet.get_layer('glbStd').trainable = False
    manTraNet.get_layer('cLSTM').trainable = False
    manTraNet.get_layer('pred').trainable = False

# ManTraNet Architecture
print(manTraNet.summary(line_length=120))

# Image Manipulation Classification Network
IMCFeatex = manTraNet.get_layer('Featex')
print(IMCFeatex.summary(line_length=120))


'''
Train on CASIA V2 Dataset
'''

optimizer = keras.optimizers.Adam(lr=lr)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                              patience=10, min_lr=0.001)

early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=20,
                                           restore_best_weights=True)

checkpointer_epoch = keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "ckpt_ft13_{epoch:03d}.hd5"),
                                                     monitor='loss',
                                                     save_best_only=True,
                                                     verbose=1,
                                                     period=1)

if wtd_bi_ce:
    loss = create_weighted_binary_crossentropy(loss_wt[0], loss_wt[1])
else:
    loss = 'binary_crossentropy'

manTraNet.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

X = HDF5Matrix('../../data/CASIA/CASIA-CMFD-Pos.hd5', 'X')
Y = HDF5Matrix('../../data/CASIA/CASIA-CMFD-Pos.hd5', 'Y')
CASIA_orig = X.data + np.array([103.939, 116.779, 123.68]).reshape([1, 1, 1, 3])    # add back RGB means
CASIA_orig = CASIA_orig[:1000, :, :, :]
CASIA_forged = Y.data[:1000, :, :, 0][:, :, :, np.newaxis]

manTraNet.fit(CASIA_orig, CASIA_forged,
           batch_size=batch_size,
           epochs=epochs,
           verbose=1,
           callbacks=[reduce_lr, early_stop, checkpointer_epoch])

# manTraNet.fit(CASIA_orig, CASIA_forged,
#            batch_size=batch_size,
#            epochs=epochs,
#            verbose=1,
#            callbacks=[checkpointer_epoch])
