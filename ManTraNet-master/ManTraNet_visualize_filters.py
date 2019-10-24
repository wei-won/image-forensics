import os
import numpy as np
import sys
from matplotlib import pyplot as plt


manTraNet_root = './'
manTraNet_srcDir = os.path.join(manTraNet_root, 'src')
sys.path.insert(0, manTraNet_srcDir)
import modelCore
manTraNet_modelDir = os.path.join(manTraNet_root, 'pretrained_weights')
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

num_layers = 2
filter_outputs = [layer.output for layer in manTraNet.get_layer('Featex').layers[:num_layers]]

filter_model = IMCFeatex.get_layer(index=1)
srm_filter = filter_model.srm_kernel
bayar_filter = filter_model.bayar_kernel

filter_wts = IMCFeatex.layers[1].get_weights()
bayar_wts = filter_wts[1]


def get_srm_list():
    # srm kernel 1
    srm1 = np.zeros([5, 5]).astype('float32')
    srm1[1:-1, 1:-1] = np.array([[-1, 2, -1],
                                 [2, -4, 2],
                                 [-1, 2, -1]])
    srm1 /= 4.
    # srm kernel 2
    srm2 = np.array([[-1, 2, -2, 2, -1],
                     [2, -6, 8, -6, 2],
                     [-2, 8, -12, 8, -2],
                     [2, -6, 8, -6, 2],
                     [-1, 2, -2, 2, -1]]).astype('float32')
    srm2 /= 12.
    # srm kernel 3
    srm3 = np.zeros([5, 5]).astype('float32')
    srm3[2, 1:-1] = np.array([1, -2, 1])
    srm3 /= 2.
    return [srm1, srm2, srm3]


def build_SRM_kernel():
    kernel = []
    srm_list = get_srm_list()
    for idx, srm in enumerate(srm_list):
        for ch in range(3):
            this_ch_kernel = np.zeros([5, 5, 3]).astype('float32')
            this_ch_kernel[:, :, ch] = srm
            kernel.append(this_ch_kernel)
    srm_kernel = np.stack(kernel, axis=-1)
    return srm_kernel


srm_wts = build_SRM_kernel()


def plot_filters(filters, plotname):
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = filters.shape[3], 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    plt.savefig(plotname)


plot_filters(bayar_wts, 'bayar.png')
plot_filters(srm_wts, 'srm.png')


# Model(inputs=manTraNet.input,
#                      outputs=filter_outputs)
#
# print(filter_model.summary(line_length=120))



