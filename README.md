# Handwritten-Digit-Classification

This repository contains kNN, naive Bayes and conditional Gaussian classifiers to label images of handwritten digits using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Each image is 8 x 8 pixels and is represented as a vector of dimension 64 by listing all the pixel values in raster scan order. The images are grayscale and the pixel values are between 0 and 1. 

There are 700 training cases and 400 test cases for each digit; they can be found in `a2digits.zip`. 

`data.py`loads data from a given zipfile, directory, and digits pixels from a given test/train set.

### RUNNING THE CODE:

For loading and plotting the MNIST dataset, run `load_and_plot.py`.

For training and evaluating the kNN classifier, run `kNN.py`.

For training and evaluating the conditional Gaussian classifier, run `conditional_gaussian.py`.

For training and evaluating the Naive Bayes classifier, run `naive_bayes.py`.

### Description of code implementation:

#### 1) kNN classifier

#### 2) Conditional Gaussian classifier

#### 3) Naive Bayes classifier

* Convert the real-valued features x into binary features b using 0.5 as a threshold: bj = 1 if xj > 0.5 otherwise bj = 0.


