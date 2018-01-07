
# coding: utf-8

# # Lab 1: MNIST Data Loader
#
# This notebook is the first lab of the "Deep Learning Explained" course.  It is derived
# from  the tutorial numbered CNTK_103A in the CNTK repository.  This notebook is used to
# download and pre-process the [MNIST][] digit images to be used for building different
# models to recognize handwritten digits.
#
# ** Note: ** This notebook must be run to completion before the other course notebooks can be run.
#
# [MNIST]: http://yann.lecun.com/exdb/mnist/

# In[1]:


# Import the relevant modules to be used later
import numpy as np
import os
import shutil
import struct
import sys
import h5py


# ## Data download
#
# We will download the data onto the local machine. The MNIST database is a standard set of
# handwritten digits that has been widely used for training and testing of machine learning
# algorithms. It has a training set of 60,000 images and a test set of 10,000 images with
# each image being 28 x 28 grayscale pixels. This set is easy to use visualize and
# train on any computer.

# In[2]:


# Functions to load SIGNS images and unpack into train and test set.
# - loadData reads image data and formats into a 28x28 long array
# - loadLabels reads the corresponding labels data, 1 for each image
# - load packs the downloaded image and labels data into a combined format to be read later by
#   CNTK text reader

def load_dataset(data_dir):
    train_dataset = h5py.File(os.path.join(data_dir, 'train_signs.h5'), "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(os.path.join(data_dir, 'test_signs.h5'), "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_data(data_dir):
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset(data_dir)

    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)

    print ("number of training examples = " + str(X_train.shape[1]))
    print ("number of test examples = " + str(X_test.shape[1]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))

    # **Note** that 12288 comes from $64 \times 64 \times 3$. Each image is square, 64 by 64
    # pixels, and 3 is for the RGB colors. Please make sure all these shapes make sense to
    # you before continuing.

    data = {"X_train": X_train,
            "X_test": X_test,
            "Y_train": Y_train,
            "Y_test": Y_test}
    return data


# # Save the images
#
# Save the images in a local directory. While saving the data we flatten the images
# to a vector (28x28 image pixels becomes an array of length 784 data points).
#
# ![mnist-input](https://www.cntk.ai/jup/cntk103a_MNIST_input.png)
#
# The labels are encoded as [1-hot][] encoding (label of 3 with 10 digits becomes
# `0001000000`, where the first index corresponds to digit `0` and the last one corresponds to digit `9`.
#
# ![mnist-label](https://www.cntk.ai/jup/cntk103a_onehot.png)
#
# [1-hot]: https://en.wikipedia.org/wiki/One-hot

# In[5]:


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


# Save the data files into a format compatible with CNTK text reader
def savetxt(filename, features, labels):
    dir = os.path.dirname(filename)

    if not os.path.exists(dir):
        os.makedirs(dir)

    print("Saving", filename)
    with open(filename, 'w') as f:
        for i in range(features.shape[1]):
            feature_str = ' '.join(map(str, features[:, i]))
            label_str = ' '.join(map(str, labels[:, i]))
            f.write('|labels {} |features {}\n'.format(label_str, feature_str))


# In[6]:

def create_datasets(data_dir):
    data = load_data(data_dir)

    print ('Writing train text file...')
    savetxt(os.path.join(data_dir, "Train-hands-64x64x3-cntk.txt"), data["X_train"], data["Y_train"])

    print ('Writing test text file...')
    savetxt(os.path.join(data_dir, "Test-hands-64x64x3-cntk.txt"), data["X_test"], data["Y_test"])

    print('Done')


# **Optional: Suggested Explorations**
#
# One can do data manipulations to improve the performance of a machine learning system. I suggest you first use the data generated so far and complete Lab 2- 4 labs. Once you have a baseline with classifying the data in its original form, now use the different data manipulation techniques to further improve the model.
#
# There are several ways data alterations can be performed. CNTK readers automate a lot of these actions for you. However, to get a feel for how these transforms can impact training and test accuracies, I strongly encourage individuals to try one or more of data perturbation.
#
# - Shuffle the training data rows to create a different set of training images.  Be sure to shuffle each image in the same way.   Hint: Use `permute_indices = np.random.permutation(train.shape[0])`. Then run Lab 2-4 with this newly permuted data.
# - Adding noise to the data can often improve (lower) the [generalization error][]. You can augment the training set by adding  noise (generated with numpy, hint: use `numpy.random`) to the training images.
# - Distort the images with [affine transformation][] (translations or rotations)
#
# [generalization error]: https://en.wikipedia.org/wiki/Generalization_error
# [affine transformation]: https://en.wikipedia.org/wiki/Affine_transformation
#

if __name__ == "__main__":
    create_datasets("../datasets")
