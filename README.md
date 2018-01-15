## Counting sea lions with Deep Neural Networks 

The purpose of this project is to Deep Learning [1] approaches to count objects from optical images 
(in contrast to SAR images, MRI images in medical imagery, ...).

Here, I applied a well-known model, already applied in biology to counts cells, **Count-ception** network [2]. 

I proposed an implementation with Keras [3], applied to count sea lion from aerial images.
The dataset comes from a Kaggle competition: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/ 


This project was done in the context of the "Introduction to Teledetection" course at IMT Atlantique (France, Brest) [4].


### The dataset: NOAA Fisheries Steller Sea Lion Population Count

The dataset comes from National Oceanic and Atmospheric Administration (NOAA), released for a Kaggle contest in 2017. 

The goal was to create algorithms that count sea lions on large aerial images(more than 3500x3500 pixels).
Sea lions were labeled into 5 categories: adult males (also known as bulls), subadult males, adult females, juveniles, and pups.
The training set was formed of about 1000 raw images.For each image, in CSV file we have the count for each category of sea lions.
'Dotted' images were released: these images are the optical images, plus colored dots.
Each color of dots represents one specific category. So we can get easily the localizations of specimens on images. See `Image1`.

![Alt text](imgs/example.png?raw=true "Image1 : dotted image with sea lions")


But we just need to count the population, not to give the exact localization of sea lions.
The localization will help us during the training phase to infer the exact number of sea lions.  

**Note**: In this work, I developed algorithms that count sea lion regardless of the categories! 

### Model : Count-ception 

I decided to use the `Count-ception` model for this problem. 

This neural architecture uses major components : 
- **Inception modules** from the GoogleNet model [5]: using 1x1 and 3x3 convolutions in parallel in the network.

- **Fully convolutional networks** [6]. Classical Convolutional Neural Networks, for image classification,
have dense layers at the end of the network. These layers make the predictions with the features extracted
from convolutional layers.  This kind of architecture works on fixed-size images: if we train the CNN on 256x256 images,
it only works on 256x256 images in the future. In fully convolutional networks,
the final feature maps (outputs of the last convolutional layer) are averaged spatially,
to make the network independent of the size of input images. See `Image2`

![Alt text](imgs/full_conv_net.png?raw=true "Image2 : Fully convolutional neural networks. Source : [6]")

Here I used the model described in the paper: only made of convolutional layers, batch normalization layers,
and LeaykyReLU layers (non-linear activation function). 

See `count_all_sea_lions.ipynb` notebook for more details about the network architecture. 

The goal of this network is to predict count maps, where we can infer the exact count of objects. 

In the count-ception approach , a full conv neural network is trained to predict **count-maps**. 
These maps are computed from dotted images. The construction is based on redundant counts so that, the count-maps give
the exact the number of sea lions present in the area and it contains imformation about the local counts (within 32x32 patches).


See paper for more details.

### Data preparation & Training

See `explore_data.ipynb`. 

So the count-ception is trained to predict the count maps from the optical images. 

![Alt text](imgs/examples.png?raw=true "Image3 : example of optical image with dots")

![Alt text](imgs/dots.png?raw=true "Image4 : dots from Image3.")

Here, I followed the model described in the paper. I split images into 256x256 patches,
from which I compute the count maps from the dotted images (count maps = target images). 



Many patches do not contain sea lions (where all the count map is zero). I kept only 2% of these! 
Images and target images are stored into HDF5 files. This file format is used to store
large numerical arrays (like Numpy arrays in python). I added a python generator that yields batches of samples 
during the training, directly from the generated HDF5 files. This tool provides an effective to way to deal with 
very large datasets (that do not fit into memory)! See `count_all_sea_lions.ipynb` for details.  

![Alt text](imgs/train_data.png?raw=true "Image5 : some training samples with the count-maps")


### Results

Some predictions : 

![Alt text](imgs/predictions.png?raw=true "Image6 : some training samples with the count-maps")



### References

- [1]: Yann LeCun, Yoshua Bengio & Geoffrey Hinton. _Deep Learning_. 2015.
- [2]: Joseph Paul Cohen et al. _Count-ception: Counting by Convolutional Redundant Counting_. Arxiv : 1703.08710. 2017
- [3]: François Chollet. _Keras_. https://github.com/keras-team/keras.
- [4]: Frederic Maussang. _Introduction to Teledetection_. IMT Atlantique. 2017.
- [5]: Christian Szegedyetal. _Going Deeper with Convolutions_. CVPR 2015.
- [6]: J. Long et al. _Fully convolutional networks for semantic segmentation_. CVPR 2015.  