# Ultrasonic endoscope recognition by Convolution neural network
## Overview
### Data
The dataset is from three different patients' endoscopic ultrasound process. I used two of them to build training set and the other as testing set. There are 5 different stages: Preparation & Finishing, Insertion, Diagonse, Needle and Dopller.<br> 
Training set contains 5000 images which have 2500 pixel labeled images by labelme and 2500 unlabeled original images. Each stages have 1000 images.<br>
Testing set contains 1000 images which are choosed from the third patient, each of the stages have 200 images.<br>
In HDF5_generate.py, I generate hdf5 files using training and testing dataset. You can generate any size of images by this class but 224,224,3 is what I used to train and test.
### Model
I used Model.py to define a model and use it to train the model and test it. The default training epochs is 20, you can change it as what you want.
### Result
The training accuracy of Dense layer is: 0.2000, with test acc is: 0.2000<br>
The training accuracy of CNN layer is: 0.9000, with test acc is: 0.6950<br>
The training accuracy of ResNet50 layer is: 0.9986, with test acc is: 0.9030<br>
The training accuracy of ResNet101 layer is: 0.7928, with test acc is: 0.7100<br>
The training accuracy of ResNet152 layer is: 0.5922, with test acc is: 0.3630<br>
## How to use
### library
Tensorflow ==> 1.15.0<br>
keras ==> 2.2.4<br>
python ==> 3.7<br>
### Run main.py
You can change the variables in main.py with the explain by comments.# PHAS0077-Xiaotian_Zhu
