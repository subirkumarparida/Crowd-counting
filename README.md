# GNR-638: Course Project

Problem Statement: Crowd counting using deep convolutional neural networks in sparse and densely congested scenarios.

Impleamentation of "CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1091–1100, 2018."

# Dataset

Shanghai Tech dataset consisting of two sub-divided datasets: Shanghai_Tech_part_A (densely populated crowd), and Shanghai_Tech_part_B (sparsely populated crowd)

# Data Preprocessing

The primary task is to convert the ground truth provided by the Shanghai Tech dataset into density maps. For a given image the dataset provided a sparse matrix consisting of the head annotations in that image. This sparse matrix was converted into a 2D density map by passing through a Gaussian Filter. The sum of all the cells in the density map results in the actual count of people in that particular image. 

# Model

The CSRNet model uses Convolutional Neural Networks to map the input image to it's respective density map. The model does not make use of any fully connected layers and thus the size of the input image is variable. The model architecture is divided into two parts, front-end and back-end. The front-end consists of 13 pretrained layers of the VGG16 model (10 Convolution layers and 3 MaxPooling layers). The fully connected layers of the VGG16 are not taken. The back-end comprises of Dilated Convolution layers. The dilation rate at which maximum accuracy was obtained was experimentally found out be 2 as suggested in the CSRNet paper.

# Results

Shanghai_Tech_part_A MAE: 250
Shanghai_Tech_part_B MAE: 65.25
