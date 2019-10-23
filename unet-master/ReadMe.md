# U-Net

This architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

### introduction
This folder contains the scripts and pre-trained model for U-Net. The U-Net was trained on CASIA dataset for image forensics problem.

### Folder Content
This folder contains the following things:

- model_unet.py	- contains custom keras-tensorflow layers, activations and model architectures
- unet_train.py - train U-Net on CASIA v2 dataset (1000/1313 images)
- unet_test.py - test on CASIA v2 dataset (313/1313 images)- FCN_train_test_onCASIA.py - train FCN on CASIA v2 dataset and show qualitative results
- unet_casia2.hdf5 - the pre-trained U-Net model
- LICENSE - the license of the original implementation for medical image segmentation, which can be found from [Implementation of deep learning framework -- Unet, using Keras](https://github.com/zhixuhao/unet)
- ReadMe.md - This file

### Instruction
To train the model, run unet_train.py

To test the model, run unet_test.py

### Dependencies
U-Net was written in Keras with the TensorFlow backend.

The model was trained and tested with
- Keras: 2.2.0
- TensorFlow: 1.12.0
- python: 3.6

We also test the repository with
- Keras: 2.2.4
- TensorFlow: 1.13.1
- python: 3.6