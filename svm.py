import numpy as np
from sklearn.svm import LinearSVC

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation
    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    clf = LinearSVC(C = 1, random_state = 0)    
    clf.fit(train_x,train_y)
    return clf.predict(test_x)      

    # C = regularization parameter, random number generator to use when shuffling the data for the dual coordinate descent (if dual=True), C = 1 by default
    # dual = True by default (see duality of Linear programming for more details on dual)
    # clf.fit(train_x,train_y) => fits model according to given trianing data
    # clf.predict(test_x) = pred_test_y


def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    clf = LinearSVC(C = 1, random_state = 0)    
    clf.fit(train_x,train_y)
    return clf.predict(test_x)   
    # the default argument for multi_class in LinearSVC is ovr


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)

