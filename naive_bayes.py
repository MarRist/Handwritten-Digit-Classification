
'''
Naive Bayes classifier implemention.
'''

import data
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5 based on the pixel_values
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)


def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data
    and return a numpy array of shape (10, 64)
    where the i-th row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    const1 = np.ones((1,64))
    const2 = 2*const1
    data_df = pd.DataFrame(train_data) 
    data_df["TrainLabels"] = train_labels
    
    # Compute sum of ones per class
    sums_df = data_df.groupby('TrainLabels', as_index=False).sum() 
    counts_df = data_df.groupby('TrainLabels', as_index=False).count() # number of datapoints per class 
    
    for digit_class in range(10):
        data_class_sums = sums_df.loc[sums_df.TrainLabels == digit_class] 
        data_class_sums = np.array(data_class_sums.drop(['TrainLabels'], axis=1))
        
        data_class_counts = counts_df.loc[counts_df.TrainLabels == digit_class]
        data_class_counts = np.array(data_class_counts.drop(['TrainLabels'], axis=1))
        
        # eta = (N_ones + 1)/N_k + 2
        eta[digit_class] = (data_class_sums + const1)/(data_class_counts + const2) 
    return eta


def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    class_images_list = []
    for i in range(0, 10):
        class_images_list.append((class_images[i]).reshape(8,8))

    # Plot all means on same axis
    plt.figure(figsize = (20, 5))
    plt.axis('off')
    all_concat = np.concatenate(class_images_list, 1)
    plt.imshow(all_concat, cmap = 'gray')
    plt.show()
    
    
def generate_new_data(eta):
    '''
    Sample a new data point from the generative distribution p(x|y,theta) for
    each value of y in the range 0...10 and plot these values
    '''
    generated_data = np.zeros((10, 64))
    generated_datum = np.zeros((64, 1))
    for idx, eta_class_pixel_prob in enumerate(eta):
        generated_data[idx] = bernoulli.rvs(eta_class_pixel_prob) # generate a random sample from a bernoulli distribution

    plot_images(generated_data)
    
    
def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    and return an n x 10 numpy array .
    '''
    n = bin_digits.shape[0]
    d = bin_digits.shape[1]
    log_gen_likelihood = np.zeros((n, 10))
    gen_likelihood = np.zeros((10, 1))
    ones = np.ones((1, d))
    
    for idx, digit in enumerate(bin_digits):
        for idx_class, eta_class in enumerate(eta):
            # compute the part when p(b=1)=eta
            digit_logEta_1 = digit.dot(np.transpose(np.log(eta_class))) 
            
            # compute the part when p(b=0)=(1-eta)
            digit_logEta_0 =  (ones - digit).dot(np.transpose(np.log(ones - eta_class))) 
            
            gen_likelihood[idx_class] = digit_logEta_1 + digit_logEta_0
            
        log_gen_likelihood[idx] = np.transpose(gen_likelihood)
        
    return log_gen_likelihood


def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    and return a be a numpy array of shape (n, 10)
    where n is the number of datapoints and 10 corresponds to each digit class.
    '''
    prior = 1/float(10) # prior for each class digit
    n = bin_digits.shape[0]
    d = bin_digits.shape[1]
    log_cond_likelihood = np.zeros((n, 10))
    gen_evidence = np.zeros((n, 1))
    
    log_gen_likelihood = generative_likelihood(bin_digits, eta)
    
    # Evidence computation (law of total probability used)
    for idx, gen_likelihood_digit in enumerate(np.exp(log_gen_likelihood)):
        gen_evidence[idx] = np.sum(gen_likelihood_digit)*prior
        
    gen_evidence_matrix = np.tile(gen_evidence, (1,10)).reshape(n,10)
    
    log_cond_likelihood = log_gen_likelihood + np.log(np.tile(prior,(n, 10))) - np.log(gen_evidence_matrix)
    
    return log_cond_likelihood



def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    n = bin_digits.shape[0]
    true_class_cond_likelihood = [] # used to store the true classes
    
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    cond_likelihood_df = pd.DataFrame(cond_likelihood) 

    # Extract the log-likelihood on the position of where the true label is stored
    for idx, true_label in enumerate(labels):
        true_class_cond_likelihood.append(cond_likelihood_df.loc[idx, true_label]) 
          
    return (1/float(n)*sum(true_class_cond_likelihood))


def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class.
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta) # (n,10) shape
    cond_likelihood_df = pd.DataFrame(np.exp(cond_likelihood))
    
    # Extracting the most probable class by finding the max probability 
    max_likelihood_class = cond_likelihood_df.idxmax(axis=1) # returns the column index of the max value per row
    
    return max_likelihood_class


def accuracy(classification_labels, true_labels):
    '''
    Compute the accuracy.
    '''
    difference = abs(np.subtract(np.array(classification_labels), true_labels))
    return 100*((classification_labels.shape[0]-np.count_nonzero(difference))/classification_labels.shape[0])



def main():
    # Load data
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Plot the model parameters
    plot_images(eta)
    
    # Compute average conditional log likelihood for train and test data
    avg_conditional_likelihood_TRAIN = avg_conditional_likelihood(train_data, train_labels, eta)
    print("The average conditional log-likelihood for TRAIN data:", avg_conditional_likelihood_TRAIN)
    
    avg_conditional_likelihood_TEST = avg_conditional_likelihood(test_data, test_labels, eta)
    print("The average conditional log-likelihood for TEST data:", avg_conditional_likelihood_TEST)
    
    # Compute classification accuracy for train and test data
    classification_train = classify_data(train_data, eta)
    accuracy_train = accuracy(classification_train, train_labels)
    print("The Naive Bayes classification accuracy on the TRAIN data is:", accuracy_train)
    
    classification_test = classify_data(test_data, eta)
    accuracy_test = accuracy(classification_test, test_labels)
    print("The Naive Bayes classification accuracy on the TEST data is:", accuracy_test)

    # Sample new datapoints per class and plot them
    generate_new_data(eta)


if __name__ == '__main__':
    main()

