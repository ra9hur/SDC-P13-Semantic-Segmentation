# Semantic Segmentation
### Introduction
The objective of the project is to label the pixels of a road in images as either road/not-road by implementing a Fully Convolutional Network for semantic segmentation. This classification will help other systems in the car determine where the free space is. This technique could be extended to more classes like road, vehicle, bicycle and pedestrian.

In this implementation, MobileNet is used for a series of convolutional layers to extract features (encoder).


### A bit about Fully Convolutional Network
Traditional convolutional network consists of series of convolutional layers, flattening convolutional layers followed by fully connected layers and ultimately a softmax activation function for classification. Flattening the convolutional layer destroys the spatial information. Hence, these networks are not capable of locating objects in an image. 

Fully convolutional network (FCN) retains the spatial information throughout the network. Output is of the same size as the input image. This is done by,

- replacing fully connected layers by 1x1 convolutional layers
- upsampling through transpose convolutional layers
- using skip connections

Structurally, an FCN comprises of two parts; encoder and decoder.

- The encoder is a series of convolutional layers like VGG and ResNet. The goal of the encoder is to extract features from the image.
- The decoder upscales the output of the encoder such that it is the same size as the original image.


### Setup

##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.


### How to run

Project files include: 
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Training data in `data` folder
 - Trained model is saved in `models` folder
 - Newest inference images from `runs` folder  (**all images from the most recent run**)

Run the following command to run the project:
```
python main.py
```

### Model Documentation

##### Approach
The project guideline reccomends to use pre-trained VGG for the encoder. The download is a huge file and is about ~ 550 MB. The project is implemented on a laptop with 4GB RAM, 2GB GPU (nvidia geforce 840M) and Ubuntu 16.04. Considering the hardware constraints, it was not feasible to load VGG weights. Instead, MobileNet (~ 70 MB) is used as an alternative. 

MobileNets are small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. MobileNets are not usually as accurate as the bigger, more resource-intensive networks. But finding that resource/accuracy trade-off is where MobileNets really shine.

##### Training Parameters

- Number of epochs: 40
- Optimizer: Adam
- Generalization:
    - Data augmentation
    - Batch normalization - did not yield good results
    - L2 regularisation with weight decay of 0.0004

##### Model Graph

![sem_seg_graph](https://user-images.githubusercontent.com/17127066/32052087-b3b3955e-ba74-11e7-9979-8244e3be4f06.png)

### Video Implementation

Developed pipeline is applied to generate a video. The project demo can be found at Youtube here.

[![semantic_segmentation](https://user-images.githubusercontent.com/17127066/31305361-b0feafd4-ab55-11e7-891c-afc6da568bec.png)](https://www.youtube.com/watch?v=I4aXmthGsys)
