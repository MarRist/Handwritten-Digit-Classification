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

* Using the binary features b and the class labels, train a Bernoulli Naive Bayes classifier using MAP estimation with prior Beta(α, β) with α = β = 2. Fit the model:

![eq1](https://latex.codecogs.com/gif.latex?p%28y%20%3D%20k%29%20%3D%201/10)
![eq2](https://latex.codecogs.com/gif.latex?p%28b_j%20%3D%201%7Cy%20%3D%20k%29%20%3D%20n_%7Bkj%7D)
![eq3](https://latex.codecogs.com/gif.latex?p%28b%7Cy%20%3D%20k%2C%20n%29%20%3D%20%5Cprod%5E%7Bd%7D_%7Bj%3D1%7D%28n_%7Bkj%7D%29%5E%7Bb_j%7D%281%20-%20n_%7Bkj%7D%29%5E%7B%281%20-%20b_j%29%7D)
![eq4](https://latex.codecogs.com/gif.latex?P%28n_%7Bkj%7D%29%20%3D%20Beta%282%2C%202%29)



