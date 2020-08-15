from HDF5_generate import HDF5_write
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from Model import Model
import random
# Show the version of tensorflow and keras.
# The version of keras I used is 2.2.4.
# The version of tensorflow I used is1.15.0.

# Generating different dataset.
unlabeled_train_set = HDF5_write('/Users/zhuxiaotian/Desktop/Project/Train', 'USFrames-train-unlabeled224', (2500, 224, 224, 3))
unlabeled_train_set.Write_it(kind_of_img='unlabeled')

unlabeled_test_set = HDF5_write('/Users/zhuxiaotian/Desktop/Project/Test', 'USFrames-test224', (1000, 224, 224, 3))
unlabeled_test_set.Write_it(kind_of_img='unlabeled')

labeled_train_set = HDF5_write('/Users/zhuxiaotian/Desktop/Project/Train_labeled', 'USFrames-train-labeled224', (5000, 224, 224, 3))
labeled_train_set.Write_it(kind_of_img='labeled')

# Loading training set and testing set.
train_set_l = h5py.File('USFrames-train-labeled224.hdf5', 'r')

test_set_unl = h5py.File('USFrames-test224.hdf5', 'r')

trian_set_unl = h5py.File('USFrames-train-unlabeled224.hdf5', 'r')

# Transfer the information into numpy format.
train_images_l = np.array(train_set_l['images'])
train_labels_l = np.array(train_set_l['labels'])

test_images_unl = np.array(test_set_unl['images'])
test_labels_unl = np.array(test_set_unl['labels'])

train_images_unl = np.array(trian_set_unl['images'])
train_labels_unl = np.array(trian_set_unl['labels'])

# Normalize the pixels' value into 0 to 1.
train_images_l /= 255.0
test_images_unl /= 255.0
train_images_unl /= 255.0

# Shuffle the testing set.
random.seed(42)
random.shuffle(test_images_unl)
random.seed(42)
random.shuffle(test_labels_unl)

# Show the images in the training set and testing set.
image_1 = train_images_l[-1]
print('This image belongs to group:', train_labels_l[-1])
plt.imshow(image_1[:,:,0])

image_2 = test_images_unl[-1]
print('This image belongs to group:', test_labels_unl[-1])
plt.imshow(image_2[:,:,0])

image_3 = train_images_unl[-1]
print('This image belongs to group:', train_labels_unl[-1])
plt.imshow(image_3[:,:,0])

# Perform the model with Simple network.
Simple = Model()
Simple.Choose_model('Simple', (224,224,3))
Simple.compile_it()
Simple.fit_it(train_images_l, train_labels_l)
Simple.get_confusion(test_images_unl, test_labels_unl)

# Perform the model with CNN network.
CNN = Model()
CNN.Choose_model('CNN', (224,224,3))
CNN.compile_it()
CNN.fit_it(train_images_l, train_labels_l)
CNN.get_confusion(test_images_unl, test_labels_unl)

# Perform the model with Resnet.
Resnet = Model()
Resnet.Choose_model('Resnet', (224,224,3))
Resnet.compile_it()
Resnet.fit_it(train_images_l, train_labels_l)
Resnet.get_confusion(test_images_unl, test_labels_unl)