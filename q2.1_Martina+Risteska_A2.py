
# coding: utf-8

# In[5]:


# Martina Risteska (ID: 1003421781)
'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import os
import zipfile
import pandas as pd
from sklearn.cross_validation import KFold


class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        # create a dataframe used later for a better data manipulation
        results_df = pd.DataFrame(self.train_labels) 
        results_df.columns = ["TrainLabels"]
        
        # compute the distance from the test_pints to all training points
        distances = self.l2_distance(test_point) 
        results_df["Distances"] = distances
        
        # find the k smallest distances (sorted from smallest to largest)
        neighbours = results_df.nsmallest(k, "Distances") 
        digits = neighbours.TrainLabels # extract the classes only
        
        # find how many times each class occur whitin the smallest distances
        digits_counts = digits.value_counts()
        digits_counts_max = digits_counts[digits_counts == digits_counts.max()] # extract the max values, will return several max values if they are equal
        
        # Brake Ties if two of more classes have an equal majority vote
        if (digits_counts_max.shape[0] > 1): 
            while digits_counts_max.shape[0] != 1:
                k = k - 1
                neighbours = results_df.nsmallest(k, "Distances") # recalculate distances 
                digits = neighbours.TrainLabels
                digits_counts = digits.value_counts() # find how many times a class appears across the k smallest distances calculated
                digits_counts_max = digits_counts[digits_counts == digits_counts.max()] # extract the max values, will return several max values if they are equal
            
        digit = digits_counts_max[digits_counts_max == digits_counts_max.max()]
 
        return digit.index[0]
    
    
    
def visualize(k, average_test_accuracy):
    
    fig = plt.figure(figsize=(15,10))
    font = {'size' : 15}
    plt.rc('font', **font)
    plt.title("Average Test Accuracy vs. K", fontsize = 22)
    plt.plot(k, average_test_accuracy, 'r')
    plt.grid()
    plt.xlabel("K", fontsize = 20)
    plt.ylabel("Average accuracy for 10-fold cross-validation", fontsize = 20)
    plt.show()
    
    
def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''

    list_average_test_accuracy = []
    kf_cv = KFold(train_data.shape[0], 10) #split the dataset into 10 folds using sklearn KFold (return indices)
    for k in k_range:
        list_test_accuracy = []
        
        for train_index, eval_index in kf_cv:
            knn = KNearestNeighbor(train_data[train_index], train_labels[train_index]) # train a KNN classifier using the train data
            eval_accuracy = classification_accuracy(knn, k, train_data[eval_index], train_labels[eval_index]) # evaluate using the validation data
            list_test_accuracy.append(eval_accuracy)
        
        list_average_test_accuracy.append(np.mean(list_test_accuracy))
    
    # vizualize the test accuracy curve and print it
    visualize(k_range, list_average_test_accuracy) 
    print("The average test accuracy per fold:", list_average_test_accuracy)
    
    optimal_k = np.argmax(list_average_test_accuracy) + 1  # returning the index of the max element in the list; the k* is + 1 since the inices are fomr 0 to 14

    return optimal_k



def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    # Accuracy: ratio of the total correct predictions out of all predictions made
    
    predicted_label = []
    
    # for each datum in the validation set get the predicted class
    for query_data_point in eval_data:
        predicted_label.append(knn.query_knn(query_data_point, k))
        
    diff = abs(np.subtract(np.array(predicted_label),eval_labels))
    
    return 100*((eval_data.shape[0]-np.count_nonzero(diff))/eval_data.shape[0])

def main():
    # Load Data
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    
    # train a KNN classifier using train set
    knn = KNearestNeighbor(train_data, train_labels) 

    # Predict labels, an example
    predicted_label = knn.query_knn(test_data[0], 5000)
    
    # Classification Accuracy for K = 1 and K = 15
    k_1 = 1
    train_accuracy_k1 = classification_accuracy(knn, k_1, train_data, train_labels)
    print("Train classification accuracy for k = 1:", train_accuracy_k1)
    test_accuracy_k1 = classification_accuracy(knn, k_1, test_data, test_labels)
    print("Test classification accuracy for k = 1:", test_accuracy_k1)
    
    k_15 = 15
    train_accuracy_k15 = classification_accuracy(knn, k_15, train_data, train_labels)
    print("Train classification accuracy for k = 15:", train_accuracy_k15)
    test_accuracy_k15 = classification_accuracy(knn, k_15, test_data, test_labels)
    print("Test classification accuracy for k = 15:", test_accuracy_k15)
    
    # Find the optimal value for k
    optimal_k = cross_validation(train_data, train_labels)
    print("The optimal K is:", optimal_k)
    
    # Classification Accuracy for optimal K
    train_accuracy_Kopt = classification_accuracy(knn, optimal_k, train_data, train_labels)
    print("Train classification accuracy for optimal k is:", train_accuracy_Kopt)
    test_accuracy_Kopt = classification_accuracy(knn, optimal_k, test_data, test_labels)
    print("Test classification accuracy for optimal k is:", test_accuracy_Kopt)
    
    
    
if __name__ == '__main__':
    main()

