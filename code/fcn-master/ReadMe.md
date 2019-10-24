# Fully Convolutional Net (FCN)

### Introduction
This folder contains the scripts and pre-trained model for FCN. The FCN was trained on CASIA dataset for image forensics problem.

The feature extractor part of this FCN utilizes the first 4 blocks of a VGG-16 network and adopts the pre-trained weights from BusterNet.

### Folder Content
This folder contains the following things:

- FullyConvNetCore.py	- contains custom keras-tensorflow layers, activations and model architectures
- FCN_train_onCASIA.py - train FCN on CASIA v2 dataset (1000/1313 images)
- FCN_test_onCASIA.py - test on CASIA v2 dataset (313/1313 images)
- FCN_train_test_onCASIA.py - train FCN on CASIA v2 dataset and show qualitative results
- pretrained_busterNet.hd5 - the pre-trained BusterNet model (only the feature extractor wis to be used)
- ReadMe.md - This file

### Instruction
To train the model, run FCN_train_test_onCASIA.py (with testing) or FCN_train_onCASIA.py (without testing)

To test the model on CASIA, run FCN_test_onCASIA.py

### Dependencies
FCN was written in Keras with the TensorFlow backend.

The model was trained and tested with
- Keras: 2.2.0
- TensorFlow: 1.12.0
- python: 3.6

We also test the repository with
- Keras: 2.2.4
- TensorFlow: 1.13.1
- python: 3.6