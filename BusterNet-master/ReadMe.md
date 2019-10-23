# BusterNet Model and Utils
This model was inspired by [BusterNet: Detecting Copy-Move Image Forgery
with Source/Target Localization](http://openaccess.thecvf.com/content_ECCV_2018/papers/Rex_Yue_Wu_BusterNet_Detecting_Copy-Move_ECCV_2018_paper.pdf).

### Introduction
This folder contains the scripts and pre-trained model for BusterNet to reproduce the results on CASIA-CMFD Dataset.

BusterNet is a deep neural architecture for image copy-move forgery detection proposed by "BusterNet: Detecting Copy-Move Image Forgery with Source/Target Localization". It features a two-branch architecture followed by a fusion module. The two branches localize potential manipulation regions via visual artifacts and copy-move regions via visual similarities, respectively.

### Folder Content
This folder contains the following things:

- BusterNetCore.py	- contains custom keras-tensorflow layers, activations and model architectures
- BusterNetUtils.py	- i/o utils and visualization
- BusterNetOnCASIA.py - test BusterNet on CASIA v2 dataset and show qualitative results
- pretrained_busterNet.hd5 - the pretrained BusterNet model
- ReadMe.md - This file

### Instruction
To test the model on CASIA-CMFD, simply run BusterNetOnCASIA.py

### Dependencies
BusterNet was written in Keras with the TensorFlow backend.

The model was trained and tested with
- Keras: 2.2.0
- TensorFlow: 1.12.0
- python: 3.6

We also test the repository with
- Keras: 2.2.4
- TensorFlow: 1.13.1
- python: 3.6