# GNR-638: Machine Learning-II for Remote Sensing - Course Project

Problem Statement: Crowd counting using deep convolutional neural networks in sparse and densely congested scenarios.

Impleamentation of "CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1091–1100, 2018."

# Dataset

Shanghai Tech dataset consisting of two sub-divided datasets: Shanghai_Tech_part_A (densely populated crowd), and Shanghai_Tech_part_B (sparsely populated crowd)

# Data Preprocessing
The primary task is to transform the ground truth provided by Shanghai Tech into density maps. A sparse matrix of head annotations was provided for each image in the dataset. By passing this sparse matrix through a Gaussian Filter, a 2D density map has been generated. Based on the density map, the actual count of people in the image is the sum of all the cells.

# Model

The CSRNet model uses Convolutional Neural Networks to map the input image to it's respective density map. The model does not make use of any fully connected layers and thus the size of the input image is variable. The model architecture is divided into two parts, front-end and back-end. The front-end consists of 13 pretrained layers of the VGG16 model (10 Convolution layers and 3 MaxPooling layers). The fully connected layers of the VGG16 are not taken. The back-end comprises of Dilated Convolution layers. The dilation rate at which maximum accuracy was obtained was experimentally found out be 2 as suggested in the CSRNet paper.

# Results

Shanghai_Tech_part_A MAE: 250

Shanghai_Tech_part_B MAE: 65.25
