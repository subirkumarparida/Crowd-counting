# GNR-638: Machine Learning-II for Remote Sensing - Course Project

### Problem Statement: Crowd counting using deep convolutional neural networks in sparse and densely congested scenarios.

Implementation of "CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1091–1100, 2018."

# File Structure

- Shanghai_Tech: The Shanghai Tech dataset consisting of part_A and part_B datasets
- CSRNet_1.ipynb: The CSRNet model trained and tested on the Shanghai_Tech part_A dataset
- CSRNet_1.ipynb: The CSRNet model trained and tested on the Shanghai_Tech part_B dataset

# Dataset

Shanghai Tech dataset consisting of two sub-datasets: Shanghai_Tech_part_A (densely populated crowd), and Shanghai_Tech_part_B (sparsely populated crowd)

# Data Pre-processing

The primary task is to transform the ground truth provided by Shanghai Tech into density maps. The target output of the dataset is given in the form of a sparse matrix of head annotations for each image in the dataset. By passing this sparse matrix through a Gaussian Filter, a 2D density map is generated. Based on the density map, the actual count of people in the image is the sum of all the cells.

# Model

The CSRNet model maps the input image to its density map using Convolutional Neural Networks. In this model, no fully connected layers are used, so the input image size can vary. The model architecture is broken up into two parts: front-end and back-end. The front end consists of 13 pre-trained layers of the VGG16 model (10 Convolution layers and 3 MaxPooling layers). The fully connected layers of the VGG16 are not used in CSRNet. The back-end comprises Dilated Convolution layers. Experimentally, the dilation rate was found to be two, at which maximum accuracy was obtained.

# Results

- #### Shanghai_Tech_part_A MAE: 250

- #### Shanghai_Tech_part_B MAE: 65.25
