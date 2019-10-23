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
from matplotlib import pyplot
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

# config = tf.ConfigProto(device_count={'GPU': 1})
# sess = tf.Session(config=config)

batch_size = 64
epochs = 400
saveImg = True
img_out_dir = 'FCN_img_out'
model_name = 'fcn_model.hd5'


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
'''
###
Changes here:
Only extract the first 100 samples.
###
'''


X_train = X[:1000, :, :, :]
X_test = X[1000:, :, :, :]
Y_train = Y.data[:1000, :, :, 0][:, :, :, np.newaxis]
Y_test = Y.data[1000:, :, :, 0][:, :, :, np.newaxis]

FCN.fit(X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[reduce_lr, early_stop])

FCN.save(model_name)


##########################
# Test on testset
##########################

FCN = load_FCN_model(model_name)
Z = FCN.predict(X_test, verbose=1)


def visualize_samples(X, Y, Z, batch_size=8, figsize=(12,4), prf_list=None, thresh=0.3, random=False):
    nb_samples = X.shape[0]
    if prf_list is None:
        if random is True:
            print("INFO: show random results")
            indices = np.random.choice( range(nb_samples), size=(batch_size,))
        else:
            print("INFO: show all results")
            indices = np.array(range(nb_samples))
    else:
        print("INFO: show random results with F1 score > {}".format( thresh ) )
        candi = np.nonzero(prf_list[:, -1] > thresh)[0].tolist()
        indices = np.random.choice(candi, size=(batch_size,))
    for idx in indices :
        # 1. add back imageNet BGR means
        x = np.array(X[idx]) + np.array([103.939, 116.779, 123.68]).reshape([1, 1, 3])
        # 2. restore image dtype and BGR->RGB
        x = np.round(x).astype('uint8')[..., ::-1]
        # 3. set gt to float
        y = np.array(Y[idx]).astype('float32')[:, :, 0]
        z = np.array(Z[idx])[:, :, 0]
        # 4. display
        pyplot.figure(figsize=figsize)
        pyplot.subplot(131)
        pyplot.imshow(x)
        pyplot.title('test image')
        pyplot.subplot(132)
        pyplot.imshow(y, cmap='gray')
        pyplot.title('ground truth')
        pyplot.subplot(133)
        pyplot.imshow(z, cmap='gray')
        pyplot.title('FCN predicted')
        # pyplot.show()
        if saveImg:
            if not os.path.exists(img_out_dir):
                os.mkdir(img_out_dir)
            img_name = str(thresh)+'_'+str(idx)+'.png'
            pyplot.savefig(os.path.join(img_out_dir, img_name))
        pyplot.close()
    return


visualize_samples(X_test, Y_test, Z, thresh=None, random=False)
