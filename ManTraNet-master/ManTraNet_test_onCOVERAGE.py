import os
import numpy as np
import cv2
import sys

from datetime import datetime
from matplotlib import pyplot
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
jpgCompress = 'jpg_80'
img_out_dir = os.path.join(img_out_root, 'img_out_COVERAGE_jpg')
if saveImg:
    if not os.path.exists(img_out_root):
        os.mkdir(img_out_root)
    if not os.path.exists(img_out_dir):
        os.mkdir(img_out_dir)
checkpoint_dir = 'ckpt'
ckpt_mdl = os.path.join(checkpoint_dir, 'ckpt_freezeFeatex.hd5')     # 'model_10.hd5'


'''
Load A Pre-trained ManTraNet Model
'''

manTraNet = modelCore.load_pretrain_model_by_index( 4, manTraNet_modelDir )
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
Load COVERAGE Data
'''

manTraNet_dataDir = os.path.join('..', '..', 'data', 'COVERAGE')
imgDir = os.path.join(manTraNet_dataDir, 'image')
jpgDir = os.path.join(manTraNet_dataDir, 'jpg_80')
mskDir = os.path.join(manTraNet_dataDir, 'mask')

filenames = os.listdir(imgDir)
L = len(filenames)/2


def get_a_pair(index, random=False):
    if random:
        idx = np.random.randint(0, L)
    else:
        idx = index
    au_filename = '{}.tif'.format(idx)
    tp_filename = '{}t.tif'.format(idx)
    jpg_filename = '{}t.jpg'.format(idx)
    tppath = os.path.join(imgDir, tp_filename)
    aupath = os.path.join(imgDir, au_filename)
    jpgpath = os.path.join(jpgDir, jpg_filename)
    gt_filename = '{}paste.tif'.format(idx)
    gtpath = os.path.join(mskDir, gt_filename)
    return tppath, aupath, gtpath, jpgpath


'''
Test on COVERAGE Data
'''


def evaluate_metrics(testy, yhat_classes):
    # compute scores for each sample
    prf_list = []
    rr = testy
    hh = yhat_classes
    ref = rr.ravel() == 255  # because the last channel is "pristine"
    hyp = hh.ravel() >= 0.2  # convert to binary
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
    # # print out results
    # print("INFO: ManTraNet Performance on COVERAGE Dataset using Pixel-Level Evaluation")
    # print("-" * 100)
    # for name, mu in zip(['Accuracy', 'Precision', 'Recall', 'F1'], prf.mean(axis=0)):
    #     print("INFO: {:>9s} = {:.3f}".format(name, mu))
    return prf


prf_list_jpg = []
prf_list_tif = []

for k in range(1, 41):
    print("Testing image # "+str(k)+"...")
    # get a sample
    forged_file, original_file, gt_file, jpg_file = get_a_pair(k, False)
    # load the original image just for reference
    ori = read_rgb_image( original_file )
    # load the ground truth mask for reference
    gt_mask = read_rgb_image( gt_file )[:,:,0]
    # manipulation detection using ManTraNet
    rgb, mask, ptime = decode_an_image_file( forged_file, manTraNet )
    jpg, mask_jpg, _ = decode_an_image_file(jpg_file, manTraNet)

    prf_tif = evaluate_metrics(gt_mask, mask)
    prf_jpg = evaluate_metrics(gt_mask, mask_jpg)
    prf_list_tif.append(prf_tif)
    prf_list_jpg.append(prf_jpg)

    if saveImg:
        pyplot.figure(figsize=(15, 10))
        pyplot.subplot(231)
        pyplot.imshow(ori)
        pyplot.title('Original Image')
        pyplot.subplot(232)
        pyplot.imshow(rgb)
        pyplot.title('Forged Image (ManTra-Net Input)')
        pyplot.subplot(233)
        pyplot.imshow(jpg)
        pyplot.title('Forged JPEG Image (ManTra-Net Input)')
        pyplot.subplot(234)
        pyplot.imshow(gt_mask)
        pyplot.title('Ground Truth Mask')
        pyplot.subplot(235)
        pyplot.imshow(mask, cmap='gray')
        pyplot.title('Predicted Mask (ManTra-Net Output)')
        pyplot.subplot(236)
        pyplot.imshow(mask_jpg, cmap='gray')
        pyplot.title('Predicted Mask for JPEG(ManTra-Net Output)')
        pyplot.suptitle(
            'Decoded {} of size {} for {:.2f} seconds'.format(os.path.basename(forged_file), rgb.shape, ptime))
        # pyplot.show()

        img_name = os.path.basename( forged_file ).split('.')[0] + '_out_' + jpgCompress + '.png'
        pyplot.savefig(os.path.join(img_out_dir, img_name))
        pyplot.close()


metrics_tif = np.array(prf_list_tif)
metrics_jpg = np.array(prf_list_jpg)
print("INFO: ManTraNet Performance on COVERAGE Dataset using Pixel-Level Evaluation")
print("-" * 100)
print("Performance on TIF:")
for name, mu in zip(['Accuracy', 'Precision', 'Recall', 'F1'], metrics_tif.mean(axis=0)[0]):
    print("INFO: {:>9s} = {:.3f}".format(name, mu))
print("Performance on JPEG:")
for name, mu in zip(['Accuracy', 'Precision', 'Recall', 'F1'], metrics_jpg.mean(axis=0)[0]):
    print("INFO: {:>9s} = {:.3f}".format(name, mu))
