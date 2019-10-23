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
from BusterNetCore import create_BusterNet_testing_model


'''
###
Changes here:
ref: https://github.com/dmlc/xgboost/issues/1715
Added the line below to avoid 'KMP duplicate lib' error. (Seems to be Mac specific problem)
Alternative solution: install 'MKL Optimizations' packages with 'conda install nomkl'.
###
'''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

saveImg = True
img_out_dir = 'img_out'

busterNetModel = create_BusterNet_testing_model('pretrained_busterNet.hd5')

X = HDF5Matrix('../../data/CASIA/CASIA-CMFD-Pos.hd5', 'X')
Y = HDF5Matrix('../../data/CASIA/CASIA-CMFD-Pos.hd5', 'Y')

'''
###
Changes here:
Only extract the first 100 samples.
###
'''
X = X.data[:100, :, :, :]
Y = Y.data[:100, :, :, :]

Z = busterNetModel.predict(X, verbose=1)


def evaluate_protocal_B(Y, Z):
    # compute scores for each sample
    prf_list = []
    for rr, hh in zip(Y, Z):
        ref = rr[..., -1].ravel() == 0 # because the last channel is "pristine"
        hyp = hh[..., -1].ravel() <= 0.5 # convert to binary
        precision, recall, fscore, _ = precision_recall_fscore_support(ref, hyp,
                                                                       pos_label=1,
                                                                       average='binary')
        prf_list.append([precision, recall, fscore])
    # stack list to an array
    prf = np.row_stack(prf_list)
    # print out results
    print("INFO: BusterNet Performance on CASIA-CMFD Dataset using Pixel-Level Evaluation Protocal-B")
    print("-" * 100)
    for name, mu in zip(['Precision', 'Recll', 'F1'], prf.mean(axis=0)):
        print("INFO: {:>9s} = {:.3f}".format(name, mu))
    return prf


prf_list = evaluate_protocal_B(Y, Z)


def check_one_sample(z, y):
    """Check BusterNet's Discernibility for one sample
    Input:
        z = np.array, BusterNet predicted mask
        y = np.array, GT mask
    Output:
        src_label = the dominant class on the src copy, if 0 then correct
        dst_label = the dominant class on the dst copy, if 1 then correct
    """

    def hist_count(arr):
        nb_src = np.sum(arr == 0)
        nb_dst = np.sum(arr == 1)
        nb_bkg = np.sum(arr == 2)
        return [nb_src, nb_dst, nb_bkg]

    def get_label(hist):
        if np.sum(hist[:2]) == 0:
            return 2
        else:
            return np.argmax(hist[:2])

    # 1. determine pixel membership from the probability map
    hyp = z.argmax(axis=-1)
    # 2. get the gt src/dst masks
    ref_src = y[..., 0] > 0.5
    ref_dst = y[..., 1] > 0.5
    # 3. count the membership histogram on src/dst masks respectively
    src_hist = hist_count(hyp[ref_src])
    src_label = get_label(src_hist)
    dst_hist = hist_count(hyp[ref_dst])
    dst_label = get_label(dst_hist)
    return src_label, dst_label


def evaluate_discernibility(Z, Y):
    lut = dict()

    '''
    ###
    Changes here:
    Initialize the LUT (look up table) with zeros to avoid missing values when calculating the precision.
    ###
    '''
    for m in range(3):
        for n in range(3):
            lut[(m,n)] = 0

    for idx, (z, y) in enumerate(zip(Z, Y)):
        src_label, dst_label = check_one_sample(z, y.astype('float32'))
        key = (src_label, dst_label)
        if key not in lut:
            lut[key] = 0
        lut[key] += 1
    # print results
    print("INFO: BusterNet's Discernibility Performance Analysis")
    print("-" * 100)

    '''
    ###
    Changes here: 
    Changed the way to sum up lut values. 
    ###
    '''
    # total = np.sum(lut.values())
    total = 0
    for key in lut:
        total = total + lut[key]

    print("{:<12s} = {}".format('Total', total))
    print("{:<12s} = {}".format('Miss', np.sum([lut[(2, k)] for k in range(3)] \
                                               + [lut[(k, 2)] for k in range(3)]) \
                                - lut[2, 2]))
    print("{:<12s} = {}".format('OptOut', lut[(1, 1)] + lut[(0, 0)]))
    print("{:<12s} = {}".format('OptIn', lut[(0, 1)] + lut[(1, 0)]))
    print("{:<12s} = {}".format('Correct', lut[(0, 1)]))
    print("-" * 100)
    print("{:<12s} = {:.3f}".format('Overall-Acc.', float(lut[(0, 1)]) / total))
    print("{:<12s} = {:.3f}".format('OptIn-Acc.', float(lut[(0, 1)]) / float(lut[(0, 1)] + lut[(1, 0)])))


evaluate_discernibility(Z, Y)


def visualize_random_samples(X, Y, Z, batch_size=8, figsize=(12,4), prf_list=None, thresh=0.3):
    nb_samples = X.shape[0]
    if prf_list is None :
        print("INFO: show random results")
        indices = np.random.choice( range(nb_samples), size=(batch_size,))
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
        y = np.array(Y[idx]).astype('float32')
        z = np.array(Z[idx])
        # 4. display
        pyplot.figure(figsize=figsize)
        pyplot.subplot(131)
        pyplot.imshow(x)
        pyplot.title('test image')
        pyplot.subplot(132)
        pyplot.imshow(y)
        pyplot.title('ground truth')
        pyplot.subplot(133)
        pyplot.imshow(z)
        pyplot.title('BusterNet predicted')
        # pyplot.show()
        if saveImg:
            if not os.path.exists(img_out_dir):
                os.mkdir(img_out_dir)
            img_name = str(thresh)+'_'+str(idx)+'.png'
            pyplot.savefig(os.path.join(img_out_dir, img_name))
        pyplot.close()
    return


visualize_random_samples(X, Y, Z, prf_list=prf_list, thresh=0.75)

visualize_random_samples(X, Y, Z, prf_list=prf_list, thresh=0.25)

