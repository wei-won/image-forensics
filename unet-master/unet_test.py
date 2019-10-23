import os
import numpy as np
from model_unet import *
from matplotlib import pyplot
from keras.utils.io_utils import HDF5Matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


X = HDF5Matrix('../../data/CASIA/CASIA-CMFD-Pos.hd5', 'X')
Y = HDF5Matrix('../../data/CASIA/CASIA-CMFD-Pos.hd5', 'Y')

# add back imageNet BGR means
X = X.data + np.array([103.939, 116.779, 123.68]).reshape([1, 1, 1, 3])
X = X/255

X_train = X[:1000, :, :, :]
X_test = X[1000:, :, :, :]
Y_train = Y.data[:1000, :, :, 0][:, :, :, np.newaxis]
Y_test = Y.data[1000:, :, :, 0][:, :, :, np.newaxis]


batch_size = 16
saveImg = True
img_out_dir = 'img_out'


model = unet()
model.load_weights("unet_casia2.hdf5")
results = model.predict(X_test, verbose=1)


def evaluate_metrics(testy, yhat_classes):
    # compute scores for each sample
    prf_list = []
    for rr, hh in zip(testy, yhat_classes):
        ref = rr[..., -1].ravel() == 1  # because the last channel is "pristine"
        hyp = hh[..., -1].ravel() >= 0.5  # convert to binary
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
    print("INFO: U-Net Performance on CASIA-CMFD Dataset using Pixel-Level Evaluation")
    print("-" * 100)
    for name, mu in zip(['Accuracy', 'Precision', 'Recall', 'F1'], prf.mean(axis=0)):
        print("INFO: {:>9s} = {:.3f}".format(name, mu))
    return prf


def visualize_samples(X, Y, Z, batch_size=8, figsize=(12,4), random=False):
    nb_samples = X.shape[0]

    if random is True:
        print("INFO: show random results")
        indices = np.random.choice( range(nb_samples), size=(batch_size,))
    else:
        print("INFO: show all results")
        indices = np.array(range(nb_samples))

    for idx in indices :
        # 1. scale back to 0 ~ 255
        x = np.array(X[idx]) * 255
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
        pyplot.title('U-Net predicted')
        # pyplot.show()
        if saveImg:
            if not os.path.exists(img_out_dir):
                os.mkdir(img_out_dir)
            img_name = str(idx)+'.png'
            pyplot.savefig(os.path.join(img_out_dir, img_name))
        pyplot.close()
    return


prf_list = evaluate_metrics(Y_test, results)

visualize_samples(X_test, Y_test, results, random=False)