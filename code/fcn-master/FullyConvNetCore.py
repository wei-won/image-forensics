"""
This file defines all BusterNet related custom layers
"""
from __future__ import print_function
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Layer, Input, Lambda
from keras.layers import BatchNormalization, Activation, Concatenate
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
import tensorflow as tf


def std_norm_along_chs(x):
    '''Data normalization along the channle axis
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        xn = tensor4d, same shape as x, normalized version of x
    '''
    avg = K.mean(x, axis=-1, keepdims=True)
    std = K.maximum(1e-4, K.std(x, axis=-1, keepdims=True))
    return (x - avg) / std


def BnInception(x, nb_inc=16, inc_filt_list=[(1, 1), (3, 3), (5, 5)], name='uinc'):
    '''Basic Google inception module with batch normalization
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
        nb_inc = int, number of filters in individual Conv2D
        inc_filt_list = list of kernel sizes, individual Conv2D kernel size
        name = str, name of module
    Output:
        xn = tensor4d, (n_samples, n_rows, n_cols, n_new_feats)
    '''
    uc_list = []
    for idx, ftuple in enumerate(inc_filt_list):
        uc = Conv2D(nb_inc, ftuple, activation='linear', padding='same', name=name + '_c%d' % idx)(x)
        uc_list.append(uc)
    if (len(uc_list) > 1):
        uc_merge = Concatenate(axis=-1, name=name + '_merge')(uc_list)
    else:
        uc_merge = uc_list[0]
    uc_norm = BatchNormalization(name=name + '_bn')(uc_merge)
    xn = Activation('relu', name=name + '_re')(uc_norm)
    return xn


class BilinearUpSampling2D(Layer):
    '''Custom 2x bilinear upsampling layer
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        x2 = tensor4d, (n_samples, 2*n_rows, 2*n_cols, n_feats)
    '''

    def call(self, x, mask=None):
        bsize, nb_rows, nb_cols, nb_filts = K.int_shape(x)
        new_size = tf.constant([nb_rows * 2, nb_cols * 2], dtype=tf.int32)
        return tf.image.resize_images(x, new_size, align_corners=True)

    def compute_output_shape(self, input_shape):
        bsize, nb_rows, nb_cols, nb_filts = input_shape
        return tuple([bsize, nb_rows * 2, nb_cols * 2, nb_filts])


class ResizeBack(Layer):
    '''Custom bilinear resize layer
    Resize x's spatial dimension to that of r

    Input:
        x = tensor4d, (n_samples, n_rowsX, n_colsX, n_featsX )
        r = tensor4d, (n_samples, n_rowsR, n_colsR, n_featsR )
    Output:
        xn = tensor4d, (n_samples, n_rowsR, n_colsR, n_featsX )
    '''

    def call(self, x):
        t, r = x
        new_size = [tf.shape(r)[1], tf.shape(r)[2]]
        return tf.image.resize_images(t, new_size, align_corners=True)

    def compute_output_shape(self, input_shapes):
        tshape, rshape = input_shapes
        return (tshape[0],) + rshape[1:3] + (tshape[-1],)


class Preprocess(Layer):
    """Basic preprocess layer for BusterNet

    More precisely, it does the following two things
    1) normalize input image size to (256,256) to speed up processing
    2) substract channel-wise means if necessary
    """

    def call(self, x, mask=None):
        # parse input image shape
        bsize, nb_rows, nb_cols, nb_colors = K.int_shape(x)
        if (nb_rows != 256) or (nb_cols != 256):
            # resize image if different from (256,256)
            x256 = tf.image.resize_bilinear(x,
                                            [256, 256],
                                            align_corners=True,
                                            name='resize')
        else:
            x256 = x
        # substract channel means if necessary
        if K.dtype(x) == 'float32':
            # input is not a 'uint8' image
            # assume it has already been normalized
            xout = x256
        else:
            # input is a 'uint8' image
            # substract channel-wise means
            xout = preprocess_input(x256)
        return xout

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 256, 256, 3)


def create_fcn(img_shape=(256, 256, 3),
                                    name='maniDet'):
    '''Create the manipulation branch for copy-move forgery detection
    '''
    # ---------------------------------------------------------
    # Input
    # ---------------------------------------------------------
    img_input = Input(shape=img_shape, name=name + '_in')
    # ---------------------------------------------------------
    # VGG16 Conv Featex
    # ---------------------------------------------------------
    bname = name + '_cnn'
    # Block 1
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name=bname + '_b1c1')(img_input)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name=bname + '_b1c2')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name=bname + '_b1p')(x1)
    # Block 2
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name=bname + '_b2c1')(x1)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name=bname + '_b2c2')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name=bname + '_b2p')(x2)
    # Block 3
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name=bname + '_b3c1')(x2)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name=bname + '_b3c2')(x3)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name=bname + '_b3c3')(x3)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name=bname + '_b3p')(x3)
    # Block 4
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name=bname + '_b4c1')(x3)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name=bname + '_b4c2')(x4)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name=bname + '_b4c3')(x4)
    x4 = MaxPooling2D((2, 2), strides=(2, 2), name=bname + '_b4p')(x4)
    # ---------------------------------------------------------
    # Deconvolution Network
    # ---------------------------------------------------------
    patch_list = [(1, 1), (3, 3), (5, 5)]
    bname = name + '_dconv'
    # MultiPatch Featex
    f16 = BnInception(x4, 8, patch_list, name=bname + '_mpf')
    # Deconv x2
    f32 = BilinearUpSampling2D(name=bname + '_bx2')(f16)
    dx32 = BnInception(f32, 6, patch_list, name=bname + '_dx2')
    # Deconv x4
    f64 = BilinearUpSampling2D(name=bname + '_bx4')(dx32)
    dx64 = BnInception(f64, 4, patch_list, name=bname + '_dx4')
    # Deconv x8
    f128 = BilinearUpSampling2D(name=bname + '_bx8')(dx64)
    dx128 = BnInception(f128, 2, patch_list, name=bname + '_dx8')
    # Deconv x16
    f256 = BilinearUpSampling2D(name=bname + '_bx16')(dx128)
    dx256 = BnInception(f256, 2, [(5, 5), (7, 7), (11, 11)], name=bname + '_dx16')
    # ---------------------------------------------------------
    # Output for Auxiliary Task
    # ---------------------------------------------------------
    pred_mask = Conv2D(1, (3, 3), activation='sigmoid', name=bname + '_pred_mask', padding='same')(dx256)
    # ---------------------------------------------------------
    # End to End
    # ---------------------------------------------------------
    model = Model(inputs=img_input, outputs=pred_mask, name=bname)
    return model


def load_FCN_model(weight_file=None):
    FCN = create_fcn()
    # img_raw = Input(shape=(None, None, 3), name='image_in')
    # img_in = Preprocess(name='preprocess')(img_raw)
    if weight_file is not None:
        try:
            FCN.load_weights(weight_file, by_name=True)
            print("INFO: successfully load pretrained weights from {}".format(weight_file))
        except Exception as e:
            print("INFO: fail to load pretrained weights from {} for reason: {}".format(weight_file, e))
    return FCN
