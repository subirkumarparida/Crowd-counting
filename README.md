# GNR-638: Machine Learning-II for Remote Sensing - Course Project

### Problem Statement: Crowd counting using deep convolutional neural networks in sparse and densely congested scenarios.

Implementation of "CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1091–1100, 2018."

# File Structure

- Shanghai_Tech: The Shanghai Tech dataset consisting of part_A and part_B datasets
- CSRNet_1.ipynb: The CSRNet model trained and tested on the Shanghai_Tech part_A dataset
- CSRNet_1.ipynb: The CSRNet model trained and tested on the Shanghai_Tech part_B dataset

# Dataset

Shanghai Tech dataset consisting of two sub-datasets: 
- Shanghai_Tech_part_A (densely populated crowd), and 
- Shanghai_Tech_part_B (sparsely populated crowd)

#### Sample Images:

<p align="left">
  <img src="https://user-images.githubusercontent.com/42779970/204106360-d32702be-fbd0-4485-932e-8ce4f5495795.jpg" width="280" height="280" title="Image">
  </p>
  
  <p align="left">
  <img src="https://user-images.githubusercontent.com/42779970/204106739-8c8174a2-360d-4384-9955-9a3b113840cf.jpg" width="280" height="280" title="Image">
  </p>


# Data Pre-processing

The primary task is to transform the ground truth provided by Shanghai Tech into density maps. The target output of the dataset is given in the form of a sparse matrix of head annotations for each image in the dataset. By passing this sparse matrix through a Gaussian Filter, a 2D density map is generated. Based on the density map, the actual count of people in the image is the sum of all the cells.

# Model

The CSRNet model maps the input image to its density map using Convolutional Neural Networks. In this model, no fully connected layers are used, so the input image size can vary. The model architecture is broken up into two parts: front-end and back-end. The front end consists of 13 pre-trained layers of the VGG16 model (10 Convolution layers and 3 MaxPooling layers). The fully connected layers of the VGG16 are not used in CSRNet. The back-end comprises Dilated Convolution layers. Experimentally, the dilation rate was found to be two, at which maximum accuracy was obtained.

# Results

- #### Shanghai_Tech_part_A MAE: 250

- #### Shanghai_Tech_part_B MAE: 65.25

#### The model is trained on a Nvidia GeForce GTX TITAN X 12GB RAM GPU.

- #Persons = 370
<p style="text-align:center">
  <img src="https://user-images.githubusercontent.com/42779970/204108740-c9f5fd30-96c5-455b-925f-ad2e7b7d308b.jpg" width="360" height="360" title="Image" alt="Check">
  <img src="https://user-images.githubusercontent.com/42779970/204108759-5491478a-8241-42a6-9b03-3e4912aa14b2.jpg" width="360" height="360" title="Density Map" alt="Density Map">
  </p>

- #Persons = 2256
<p align="left">
  <img src="https://user-images.githubusercontent.com/42779970/204108744-2d9a5167-866e-461f-adfe-a38e31e5cf14.jpg" width="360" height="360" title="Image">
  <img src="https://user-images.githubusercontent.com/42779970/204108765-095b8ec1-85ba-443e-9125-caecc8f59340.jpg" width="360" height="360" title="Density Map" alt="Density Map">
</p>


- #Persons = 748
<p align="left">
  <img src="https://user-images.githubusercontent.com/42779970/204108747-aca39f6f-451e-4175-8f9e-0436d891b2ce.jpg" width="360" height="360" title="Image">
  <img src="https://user-images.githubusercontent.com/42779970/204108769-ecdc2501-1c65-4f4e-8c01-2939258e3782.jpg" width="360" height="360" title="Density Map" alt="Density Map">
</p>
