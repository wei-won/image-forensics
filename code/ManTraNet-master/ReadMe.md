# ManTraNet Model
This model was inspired by and modified based on [ManTraNet: Manipulation Tracing Network For Detection And Localization of Image Forgeries With Anomalous Features](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_ManTra-Net_Manipulation_Tracing_Network_for_Detection_and_Localization_of_Image_CVPR_2019_paper.pdf).

### Introduction
This folder contains the scripts and pre-trained model for ManTraNet to reproduce the results on COVERAGE dataset and CASIA-CMFD dataset.

ManTraNet is an image forgery detection and localization architecture claimed to cover copy-move, splicing, removal and enhancement forgeries.

[Bayar filter](http://misl.ece.drexel.edu/wp-content/uploads/2017/07/Bayar_IHMMSec_2016.pdf) and [SRM filters](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Learning_Rich_Features_CVPR_2018_paper.pdf) were adopted to suppress RGB semantic contents by extracting local noise features.

### Folder Content
This folder contains the following things:

- *ManTraNet_train.py* - train ManTraNet on CASIA v2 dataset (1000/1313 images) 
- *ManTraNet_test_onCASIA.py* - test ManTraNet on CASIA v2 dataset and show qualitative results
- *ManTraNet_test_onCOVERAGE.py* - test ManTraNet on COVERAGE dataset and show qualitative results
- *ManTraNet_visualize_feature_map.py* - visualize the feature maps coming out from the highpass filters (Bayar & SRM) based on the CASIA dataset
- *ManTraNet_visualize_filters.py* - visualize the weights of highpass filters (Bayar & SRM)
- src
    - *modelCore.py* - contains custom keras-tensorflow layers, activations and model architectures
- ckpt
    - *ckpt_freezeFeatex.hd5* - model check point of training with the Manipulation Trace Feature Extractor frozen
    - *ckpt_freezeLoc.hd5* - model check point of training with the Local Anomaly Detection Network frozen
    - *model_10.hd5* - fine-tuned model for 5 epochs (on the entire network) 
- data - sample images provided by the authors
- pretrained_weights - pre-trained model provided by the authors
- *ReadMe.md* - This file

### Instruction
To train the model, run *ManTraNet_train.py*.

To test the model, run *ManTraNet_test_onCASIA.py* or *ManTraNet_test_onCOVERAGE.py* to get the performance on corresponding dataset.

### Dependencies
ManTraNet was written in Keras with the TensorFlow backend.

The model was trained and tested with
- Keras: 2.2.0
- TensorFlow: 1.12.0
- python: 3.6

We also test the repository with
- Keras: 2.2.4
- TensorFlow: 1.13.1
- python: 3.6
