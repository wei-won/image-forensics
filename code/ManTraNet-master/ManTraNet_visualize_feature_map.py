import os
import numpy as np
import cv2
import sys

from matplotlib import pyplot as plt
from keras.models import Model
from keras.utils.io_utils import HDF5Matrix


manTraNet_root = './'
manTraNet_srcDir = os.path.join(manTraNet_root, 'src')
sys.path.insert(0, manTraNet_srcDir)
import modelCore
manTraNet_modelDir = os.path.join(manTraNet_root, 'pretrained_weights')
saveImg = True
img_out_root = 'img_out'
img_out_dir = os.path.join(img_out_root, 'feat_map_CASIA')
checkpoint_dir = 'ckpt'
ckpt_mdl = os.path.join(checkpoint_dir, 'ckpt_freezeFeatex.hd5')


'''
Load A Pretrained ManTraNet Model
'''

manTraNet = modelCore.load_pretrain_model_by_index(4, manTraNet_modelDir)
manTraNet.load_weights(ckpt_mdl)

# ManTraNet Architecture
print(manTraNet.summary(line_length=120))

# Image Manipulation Classification Network
IMCFeatex = manTraNet.get_layer('Featex')
print(IMCFeatex.summary(line_length=120))


'''
Extract the high pass output
Ref: https://stackoverflow.com/questions/56147954/get-input-layer-of-keras-model-after-saving-and-re-loading-it-from-disk
'''

filter_model = Model(inputs=IMCFeatex.get_input_at(0), outputs=IMCFeatex.layers[1].output)
print(filter_model.summary(line_length=120))

filter_weights = filter_model.get_weights()


'''
Load CASIA V2 Data
'''

def read_rgb_image(image_file):
    rgb = cv2.imread(image_file, 1)[..., ::-1]
    return rgb


def extract_feature_map_from_array(rgb, model):
    x = np.expand_dims(rgb.astype('float32') / 255. * 2 - 1, axis=0)
    y = model.predict(x)
    return y


def decode_an_image_file(image_file, model):
    rgb = read_rgb_image(image_file)
    feature_maps = extract_feature_map_from_array(rgb, model)
    return rgb, feature_maps


X = HDF5Matrix('../../data/CASIA/CASIA-CMFD-Pos.hd5', 'X')
Y = HDF5Matrix('../../data/CASIA/CASIA-CMFD-Pos.hd5', 'Y')
CASIA_orig = X.data + np.array([103.939, 116.779, 123.68]).reshape([1, 1, 1, 3])
CASIA_orig = CASIA_orig[1000:, :, :, :]
CASIA_forged = Y.data[1000:, :, :, 0][:, :, :, np.newaxis]

if saveImg:
    if not os.path.exists(img_out_dir):
        os.mkdir(img_out_dir)

for k in range(CASIA_orig.shape[0]):
    img = CASIA_orig[k, ..., :3]
    img = np.round(img).astype('uint8')[..., ::-1]
    gt = CASIA_forged[k, ...].astype('float32')[:, :, 0]
    activations = extract_feature_map_from_array(img, filter_model)
    highpass_filter_activation = activations
    print(highpass_filter_activation.shape)

    # layer_names = []
    # for layer in filter_model.layers:
    #     layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    '''
    Ref: https://machinelearningmastery.com/
    how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
    '''

    col = 3

    std_conv_act = highpass_filter_activation[0, :, :, :4]
    srm_act = highpass_filter_activation[0, :, :, 4:13]
    srm_act_distinct = srm_act[:, :, [0,3,6]]
    barya_act = highpass_filter_activation[0, :, :, 13:]

    ix = 1
    for _ in range(srm_act_distinct.shape[-1]):
        # plt.figure(figsize=(15, 5))
        ax_srm = plt.subplot(int(srm_act_distinct.shape[-1]/col), col, ix)
        plt.imshow(srm_act_distinct[:, :, ix-1], cmap='gray')
        ix += 1
    if saveImg:
        maps_name = str(k) + '_srm_map.png'
        plt.savefig(os.path.join(img_out_dir, maps_name))
    plt.close()

    ix = 1
    for _ in range(barya_act.shape[-1]):
        # plt.figure(figsize=(15, 5))
        ax_barya = plt.subplot(int(barya_act.shape[-1]/col), col, ix)
        plt.imshow(barya_act[:, :, ix-1], cmap='gray')
        ix += 1
    if saveImg:
        maps_name = str(k) + '_barya_map.png'
        plt.savefig(os.path.join(img_out_dir, maps_name))
    plt.close()
