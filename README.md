# Nuclei_Segmentation_Experiments_Demo
Overlapping nuclei segmentation using Deep Interval-Masker-Aware Networks and Marker-controlled Watershed  
![H&E Image](https://github.com/appiek/Nuclei_Segmentation_Experiments_Demo/blob/master/160120_152.png?raw=true)
![Segmentation result](https://github.com/appiek/Nuclei_Segmentation_Experiments_Demo/blob/master/Slide_160120_152_fg_interval_marker_watershedrgb_result.png?raw=true)  

* Left image: the H&E stained histopathology images  
* Right image: the nuclei segmentation result using our method, in which the adjacent nuclei are labeled with different colors.

## Overview
We present a novel and efficient computing framework for segmenting the overlapping nuclei by combining Marker-controlled Watershed with our proposed convolutional neural network (DIMAN). 
We implemented our method based on the open-source machine learning framework TensorFlow  and reinforcement learning library TensorLayer.This repository contains all code used in our experiments, incuding the data preparation, model construction, model training and
result evaluation. For comparison with our method, we also utilized TensorFlow and TensorLayer to reimplement four known semantic segmentation convolutional neural networks: FCN8s, U-Net, HED and SharpMask.

## Dependencies  
* Python 3.x
* TensorFlow 1.x
* TensorLayer 1.5.4
* Scikit-image 13.0
* Numpy
* Scipy

## Dataset
We conducted the experiments on two public H&E stained histopathology image datasets: MICCAI2017 and BNS.  

* MICCAI2017:  includes totally 32 annotated image tiles [Link](http://miccai.cloudapp.net/competitions/)
* BNS: contains 33 manually annotated H&E stained histopathology
images with a total of 2754 cells [Link](https://peterjacknaylor.github.io/)  
* We have shared the two datasets by Google Drive [Link](https://drive.google.com/open?id=1-11wp70IAc-Ba-oI89qhtCa_VOXZ93Ls)

## Pre-trained model
will be provided soon!

## Composition of code
* ./
** step_1: randomly extracting the image patches from original images 
** step_2: randomly divide the image patches as training and validation data
** step_3: producing the pixel-wise weight map for solving the class-imbalance problem
** step_4: transforming the image patches into tfrecord file
** step_5: training multiple networks with same hyper-parameters
** step_6: using the networks to segment the testing images
** step_7: evaluating the segmentation results 
** step_8: arranging the evaluation data as a table

## Contact information  
* E-mail: xlpflyinsky@foxmail.com
