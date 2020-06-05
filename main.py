import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *                     ### All the functions and constants can be imported using *.
from linear_regression import *			### import uses __import__() to search for module, and if not found, it would raise ImportError
from svm import *
from softmax import *
from features import *
from kernel import *

#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################

# TODO: first fill out functions in linear_regression.py, otherwise the functions below will not work


# def run_linear_regression_on_MNIST(lambda_factor):
#     """
#     Trains linear regression, classifies test data, computes test error on test set

#     Returns:
#         Final test error
#     """
#     train_x, train_y, test_x, test_y = get_MNIST_data()
#     train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
#     test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
#     theta = closed_form(train_x_bias, train_y, lambda_factor)
#     test_error = compute_test_error_linear(test_x_bias, test_y, theta)
#     return test_error
# # Don't run this until the relevant functions in linear_regression.py have been fully implemented.

# print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=0.01))

# 	##-----------------------------------------------------------------------------------------------------------------
# 	### RESULT: 
# 	### FOR lambda_factor 1,0.1,0.01 => 0.7697,0.7698,0.7702 test_error
# 	### No matter what lambda_factor,test_error is large
# 	### The closed form solution of linear regression is the solution of optimizing the mean squared error loss.
# 	### This is not an appropriate loss function for a classification problem.
# 	##------------------------------------------------------------------------------------------------------------------


# #######################################################################
# # 3. Support Vector Machine
# #######################################################################

# # TODO: first fill out functions in svm.py, or the functions below will not work

# def run_svm_one_vs_rest_on_MNIST():
#     """
#     Trains svm, classifies test data, computes test error on test set

#     Returns:
#         Test error for the binary svm
#     """
#     train_x, train_y, test_x, test_y = get_MNIST_data()
#     train_y[train_y != 0] = 1
#     test_y[test_y != 0] = 1
#     pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
#     test_error = compute_test_error_svm(test_y, pred_test_y)
#     return test_error

# print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())
	
# 	##--------------------------------------------------------------------------------------------------------------
# 	### RESULT	
# 	### SVM one vs. rest test_error: 0.0075 for C = 0.1, 0.0085 for C = 0.9, 0.0087 for C = 1
# 	### With this implimentation of C-SVM we can interpret that more the value of C smaller will be the margin separating hyperplane
# 	### and thus smaller tolerance of violation
# 	##--------------------------------------------------------------------------------------------------------------


# def run_multiclass_svm_on_MNIST():
#     """
#     Trains svm, classifies test data, computes test error on test set
#     Returns:
#         Test error for the binary svm
#     """
#     train_x, train_y, test_x, test_y = get_MNIST_data()
#     pred_test_y = multi_class_svm(train_x, train_y, test_x)
#     test_error = compute_test_error_svm(test_y, pred_test_y)
#     return test_error


# print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################

def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function
    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150
    Saves the final theta to ./theta.pkl.gz
    Returns:
        Final test error
    """
    # n_components = 10


    train_x, train_y, test_x, test_y = get_MNIST_data()

    ###-------THIS PART OF CODE IS FOR PCA---------------
    # train_x_centered, feature_means = center_data(train_x)
    # pcs = principal_components(train_x_centered)
    # train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
    # test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)
    ##---------------------------------------------------

    ##-------THIS PART OF CODE IS FOR CUBIC KERNEL-------
    # train_pca10 = project_onto_PC(train_x, pcs, n_components, feature_means)
    # test_pca10 = project_onto_PC(test_x, pcs, n_components, feature_means)
    # train_cube = cubic_features(train_pca10)
    # test_cube = cubic_features(test_pca10)
	##---------------------------------------------------

    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    # theta = polynomial_kernel(train_x, train_y,1,3)
    # theta = rbf_kernel(train_x,train_y,gamma)

    test_error = compute_test_error(test_x, test_y, polynomial_kernel(train_x, train_y,1,3), temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")
    return test_error


print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))
	##--------------------------------------------------------------------------------------------------------------
 	### RESULT	
 	### softmax test_error= 0.1005 for temp_parameter = 1,
 	### softmax test_error= 0.084 for temp_parameter = 0.5,
 	### softmax test_error= 0.1261 for temp_parameter = 2
 	### Smaller temperature parameter means that there is less variance in our distribution, and larger temperature, more variance. 
 	### In other words smaller temperature parameter favors larger thetas, and larger temperature parameter makes the distribution more uniform.
 	##--------------------------------------------------------------------------------------------------------------


# #######################################################################
# # 6. Changing Labels
# #######################################################################

# def run_softmax_on_MNIST_mod3(temp_parameter=1):
#     """
#     Trains Softmax regression on digit (mod 3) classifications.
#     """
#     train_x, train_y, test_x, test_y = get_MNIST_data()
#     theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
#     test_error = compute_test_error_mod3(test_x, test_y, theta, temp_parameter)
#     return test_error
# print('softmax mod3 test_error = ',run_softmax_on_MNIST_mod3(temp_parameter=1))

#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##

# TODO: First fill out the PCA functions in features.py as the below code depends on them.


# n_components = 18

# ###Correction note:  the following 4 lines have been modified since release.
# train_x_centered, feature_means = center_data(train_x)
# pcs = principal_components(train_x_centered)
# train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
# test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

# # train_pca (and test_pca) is a representation of our training (and test) data
# # after projecting each example onto the first 18 principal components.


# # TODO: Train your softmax regression model using (train_pca, train_y) and evaluate its accuracy on (test_pca, test_y).
# # softmax regression model tested for PCA. 
# # RESULT: softmax test_error= 0.1474 for n_components = 18.



# # Used the plot_PC function in features.py to produce scatterplot
# # of the first 100 MNIST images, as represented in the space spanned by the first 2 principal components found above.
# plot_PC(train_x[range(000, 100), ], pcs, train_y[range(000, 100)], feature_means)		#feature_means added since release


# # Used the reconstruct_PC function in features.py to show the first and second MNIST images as reconstructed solely from
# # their 18-dimensional principal component representation. Compare the reconstructed images with the originals.
# firstimage_reconstructed = reconstruct_PC(train_pca[0, ], pcs, n_components, train_x, feature_means)		#feature_means added since release
# plot_images(firstimage_reconstructed)
# plot_images(train_x[0, ])

# secondimage_reconstructed = reconstruct_PC(train_pca[1, ], pcs, n_components, train_x, feature_means)		#feature_means added since release
# plot_images(secondimage_reconstructed)
# plot_images(train_x[1, ])


# ## Cubic Kernel ##
# # Find the 10-dimensional PCA representation of the training and test set
# # First fill out cubicFeatures() function in features.py as the below code requires it.

# train_cube = cubic_features(train_pca10)
# test_cube = cubic_features(test_pca10)	# train_cube (and test_cube) is a representation of our training (and test) data

# # after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.
# # softmax regression model tested for CUBIC KERNEL. 
# # RESULT:softmax test_error= 0.0839


# # TODO: Train your softmax regression model using (train_cube, train_y)
# #       and evaluate its accuracy on (test_cube, test_y).
