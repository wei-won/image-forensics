import os
import numpy as np
import cv2
import sys

from datetime import datetime
from matplotlib import pyplot
from keras.utils.io_utils import HDF5Matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


manTraNet_root = './'
manTraNet_srcDir = os.path.join(manTraNet_root, 'src')
sys.path.insert(0, manTraNet_srcDir)
import modelCore
manTraNet_modelDir = os.path.join(manTraNet_root, 'pretrained_weights')
saveImg = False
img_out_root = 'img_out'
img_out_dir = os.path.join(img_out_root, 'img_out_CASIA')
if saveImg:
    if not os.path.exists(img_out_root):
        os.mkdir(img_out_root)
    if not os.path.exists(img_out_dir):
        os.mkdir(img_out_dir)
checkpoint_dir = 'ckpt'
ckpt_mdl = os.path.join(checkpoint_dir, 'ckpt_ft13_049.hd5')     # 'model_10.hd5'


'''
Load A Pre-trained ManTraNet Model
'''

manTraNet = modelCore.load_pretrain_model_by_index(4, manTraNet_modelDir)
manTraNet.load_weights(ckpt_mdl)

# ManTraNet Architecture
print(manTraNet.summary(line_length=120))

# Image Manipulation Classification Network
IMCFeatex = manTraNet.get_layer('Featex')
print(IMCFeatex.summary(line_length=120))


def read_rgb_image(image_file):
    rgb = cv2.imread(image_file, 1)[..., ::-1]
    return rgb


def decode_an_image_array(rgb, manTraNet):
    x = np.expand_dims(rgb.astype('float32') / 255. * 2 - 1, axis=0)
    t0 = datetime.now()
    y = manTraNet.predict(x)[0, ..., 0]
    t1 = datetime.now()
    return y, t1 - t0


def decode_an_image_file(image_file, manTraNet):
    rgb = read_rgb_image(image_file)
    mask, ptime = decode_an_image_array(rgb, manTraNet)
    return rgb, mask, ptime.total_seconds()


'''
Load CASIA V2 Data
'''

X = HDF5Matrix('../../data/CASIA/CASIA-CMFD-Pos.hd5', 'X')
Y = HDF5Matrix('../../data/CASIA/CASIA-CMFD-Pos.hd5', 'Y')
CASIA_orig = X.data + np.array([103.939, 116.779, 123.68]).reshape([1, 1, 1, 3])
CASIA_orig = CASIA_orig[1000:, :, :, :]
CASIA_forged = Y.data[1000:, :, :, 0][:, :, :, np.newaxis]


'''
Test on CASIA V2 Data
'''

Z = []

for k in range(CASIA_orig.shape[0]):
    img = CASIA_orig[k, ..., :3]
    img = np.round(img).astype('uint8')[..., ::-1]
    gt = CASIA_forged[k, ...].astype('float32')[:, :, 0]
    mask, ptime = decode_an_image_array(img, manTraNet)
    ptime = ptime.total_seconds()
    Z.append(mask)

    if saveImg:
        pyplot.figure(figsize=(15, 5))
        pyplot.subplot(131)
        pyplot.imshow(img)
        pyplot.title('Forged Image (ManTra-Net Input)')
        pyplot.subplot(132)
        pyplot.imshow(gt, cmap='gray')
        pyplot.title('Ground Truth (Forgery Mask)')
        pyplot.subplot(133)
        pyplot.imshow(mask, cmap='gray')
        pyplot.title('Predicted Mask (ManTra-Net Output)')
        pyplot.suptitle('Decoded {} of size {} for {:.2f} seconds'.format(k, img.shape, ptime))
        # pyplot.show()
        img_name = str(k) + '.png'
        pyplot.savefig(os.path.join(img_out_dir, img_name))
        pyplot.close()


def evaluate_metrics(testy, yhat_classes):
    # compute scores for each sample
    prf_list = []
    for rr, hh in zip(testy, yhat_classes):
        ref = rr[..., -1].ravel() == 1  # because the last channel is "pristine"
        hyp = hh[..., -1].ravel() >= 0.2  # convert to binary
        acc = accuracy_score(ref, hyp)
        precision = precision_score(ref, hyp)
        recall = recall_score(ref, hyp)
        fscore = f1_score(ref, hyp)
        # precision, recall, fscore, _ = precision_recall_fscore_support(ref, hyp,
        #                                                                pos_label=1,
        #                                                                average='binary')
        prf_list.append([acc, precision, recall, fscore])
    # stack list to an array
    prf = np.row_stack(prf_list)
    # print out results
    print("INFO: ManTraNet Performance on CASIA-CMFD Dataset using Pixel-Level Evaluation")
    print("-" * 100)
    for name, mu in zip(['Accuracy', 'Precision', 'Recall', 'F1'], prf.mean(axis=0)):
        print("INFO: {:>9s} = {:.3f}".format(name, mu))
    return prf


netOut = np.array(Z)[:, :, :, np.newaxis]

prf_list = evaluate_metrics(CASIA_forged, netOut)
# print(prf_list)