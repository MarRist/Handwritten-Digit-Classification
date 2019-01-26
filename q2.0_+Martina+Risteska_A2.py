
# coding: utf-8

# In[4]:


# Martina Risteska (ID: 1003421781)
'''
Question 2.0 Skeleton Code

Here you should load the data and plot
the means for each of the digit classes.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def plot_means(train_data, train_labels):
    means = []
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i) # use the function build in data.py
        # Compute mean of class i
        mean = i_digits.sum(axis = 0)/i_digits.shape[0] #sum over columns - axis = 0, sum over rows - axis = 1
        means.append(mean.reshape(8,8))

    # Plot all means on same axis
    plt.figure(figsize=(20, 5))
    plt.axis('off')
    all_concat = np.concatenate(means, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

if __name__ == '__main__':
    train_data, train_labels, _, _ = data.load_all_data_from_zip('a2digits.zip', 'data')
    plot_means(train_data, train_labels)

