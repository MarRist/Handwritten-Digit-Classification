
'''
Implementation of Conditional Gaussian classifier.
'''

import data
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
import math

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class and return a numpy array of size (10,64)
    where the i-th row correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    data_df = pd.DataFrame(train_data) 
    data_df["TrainLabels"] = train_labels
    
    # Compute means for each of the class digit
    means_df = data_df.groupby('TrainLabels', as_index=False).mean()
    
    return np.array(means_df.drop(['TrainLabels'], axis=1))


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class and 
    return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    data_df = pd.DataFrame(train_data) 
    data_df["TrainLabels"] = train_labels
    
    # Compute means for each of the classes
    means = compute_mean_mles(train_data, train_labels)
    
    for digit_class in range(0,10):
        # slice the data for each digit class
        train_data_class = data_df.loc[data_df.TrainLabels == digit_class]
        train_data_class = train_data_class.drop(['TrainLabels'], axis=1)
        
        # create a means matrix from the means vector per class
        means_matrix = np.tile(means[digit_class], (train_data_class.shape[0],1)) 
        
        class_covariance = (train_data_class - means_matrix).T.dot((train_data_class - means_matrix))/train_data_class.shape[0]
        covariances[digit_class] = (class_covariance) 

    return covariances


def check_symmetric(a, tol=1e-8):
    '''
    Check if a matrix is symmetric.
    '''
    return np.allclose(a, a.T, atol=tol)

def is_pos_def(a):
    '''
    Check if a matrix is positive semidefinite.
    '''
    return np.all(np.linalg.eigvals(a) > 0)

def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    cov_diag = []
    for i in range(0, 10):
        cov_diag.append((np.log(np.diag(covariances[i]))).reshape(8,8))

    # Plot all means on same axis
    plt.figure(figsize=(20, 5))
    plt.axis('off')
    all_concat = np.concatenate(cov_diag, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()
    
    
def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    and return an n x 10 numpy array 
    '''
    
    d = digits.shape[1]
    n = digits.shape[0]
    log_p_digits = np.zeros((digits.shape[0], 10)) # used to store the log density for each class
    
    prior = 1/float(10)
    const = d*np.log(2*math.pi)
    
    # Compute generative log likelihood
    for idx, digit in enumerate(digits):
        digit = digit.reshape(1,d)
        log_density_datum = np.zeros((10, 1)) # used to store the log density for one datum for each class
        
        for i in range(10):
            means_k = means[i].reshape(1, d) # mean vector for a specific class k
            covariance_k = covariances[i] # slice out the covariance matrix for the corresponding class
            
            covariances_k_inverse = np.linalg.inv(covariance_k) # calculate the inverse 
            covariances_k_determinant = np.linalg.det(covariance_k) # calculate the determinant
            const_term = const + np.log(covariances_k_determinant) # calculate the const term
            log_density_datum[i] = (-0.5*(const_term + (digit - means_k).dot(covariances_k_inverse).dot((digit - means_k).T)))
        
        log_p_digits[idx] = np.transpose(log_density_datum) # store the cov matrix per class in the numpy array
        
    return log_p_digits
        
    
    

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    and return a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    ''' 
    prior = 1/float(10)
    d = digits.shape[1]
    n = digits.shape[0]
    generative_evidence = np.zeros((n, 1))
    
    # compute gen log-likelihood
    log_generative_p = generative_likelihood(digits, means, covariances)
    
    # Compute the evidence (law of total probability applied)
    for idx, generative_p_digit in enumerate(np.exp(log_generative_p)):
        generative_evidence[idx] = np.sum(generative_p_digit)*prior
     
    generative_evidence_matrix = np.tile(generative_evidence, (1,10)).reshape(n,10) # construct a matrix by repeating the evidence vector values
    
    log_conditional_p = log_generative_p + np.log(np.tile(prior,(n, 10))) - np.log(generative_evidence_matrix)
    
    return log_conditional_p


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    # digits refer to train/test data; labels refer to train/test labels
    
    true_class_cond_likelihood = [] # used to store the true class log-likelihoods
    n = digits.shape[0]
    
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    cond_likelihood_df = pd.DataFrame(cond_likelihood)
    
    for idx, true_label in enumerate(labels):
        true_class_cond_likelihood.append(cond_likelihood_df.loc[idx, true_label])
          
    return (1/float(n)*sum(true_class_cond_likelihood))


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    cond_likelihood_df = pd.DataFrame(np.exp(cond_likelihood)) # (n, 10) shape; each column indicate the digit class
    
    max_likelihood_class = cond_likelihood_df.idxmax(axis=1) # extract the column index values of the max value per row
    
    return max_likelihood_class


def accuracy(classification_labels, true_labels):
    '''
    Compute the accuracy between the predicted and the true labels 
    i.e., classification_labels and true_labels
    '''
    difference = abs(np.subtract(np.array(classification_labels), true_labels))
    
    return 100*((classification_labels.shape[0]-np.count_nonzero(difference))/classification_labels.shape[0])


def main():
    # Load data
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    
    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    
    # Check if the cov matrices are symmetric and non-negative (e.g. the first cov matrix)
    print("Is cov matric symmetric:" ,check_symmetric(covariances[0]))
    print("Is cov matrix positive semi-definite:", is_pos_def(covariances[0]))
    
    # Plot the variance for each class
    plot_cov_diagonal(covariances)

    # Calculating average conditional log-likelihood for both train and test set
    avg_conditional_likelihood_TRAIN = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    print("The average conditional log-likelihood for TRAIN data:", avg_conditional_likelihood_TRAIN)
    
    avg_conditional_likelihood_TEST = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("The average conditional log-likelihood for TEST data:", avg_conditional_likelihood_TEST)

    
    # Calculating classification accuracy for both train and test set
    classification_train = classify_data(train_data, means, covariances)
    print("Classification TRAIN accuracy for Gaussian:", accuracy(classification_train, train_labels))
    
    classification_test = classify_data(test_data, means, covariances)
    print("Classification TEST accuracy for Gaussian:", accuracy(classification_test, test_labels))

if __name__ == '__main__':
    main()

