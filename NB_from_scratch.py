import warnings
import pandas as pd
import numpy as np

import faulthandler

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
warnings.filterwarnings(action="ignore")


# Set seeds
import random 
random.seed(100)
np.random.seed(100)

TEST_SIZE  = 0.5


class NaiveBayes:

    @staticmethod
    def find_unique_classes_with_count(a):
        __class = np.asarray(np.unique(a,return_counts=True)).T
        return __class[:,0], __class[:,1]
    
    def __init__(self, alpha = 1):
        self.alpha = alpha

    # Calculate P(C1)
    def calculate_class_probabilities(self):

        for idx, cls in enumerate(self.classes):
            self.class_probabilities[cls] = self.class_freq[idx] /  self.rows
    
    # PDF of the gaussian distribution to compute stats
    def gaussian_pdf(self,x,class_idx):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


    # TODO: Refactor the code, check the correctness of data...
    def calculate_statistics(self):
        self.mean = np.zeros((len(self.classes), self.columns), dtype=np.float64)
         
        self.var = np.zeros((len(self.classes), self.columns), dtype=np.float64) 
        for idx, cls in enumerate(self.classes):
                self.mean[idx,:] = self.X_train[self.y_train == cls].mean(axis = 0)
                self.var[idx,:] = self.X_train[self.y_train == cls].var(axis = 0)


    def fit(self, X_train, y_train):
        self.X_train = X_train 
        self.y_train = y_train       
        
        self.classes, self.class_freq = NaiveBayes.find_unique_classes_with_count(y_train)
        self.rows = X_train.shape[0]
        self.columns = X_train.shape[1]


        self.class_probabilities = {}
        # Calculating the class probabilities
        self.calculate_class_probabilities()

        # Calculate the mean, var dev for Gaussian transform
        self.calculate_statistics()


    
    def _predict(self, x):
        "x: single instance of 1 row and d column size"
        diff_cls_prediction = []
        for idx, cls in enumerate(self.classes):
            prior = np.log(self.class_probabilities[cls])
            likelihood = np.sum(np.log(self.gaussian_pdf(x,idx)))
            diff_cls_prediction.append(prior+likelihood)
        
        return self.classes[np.argmax(diff_cls_prediction)]
    
    def predict(self,X_test):
        predictions = []
        for x in X_test:
            predictions.append(self._predict(x))
        return predictions



#Import the datasets as X,y and return as train and test in proportion to pre-set split ratio
def import_data():
    iris = load_breast_cancer()
    X,y = iris.data, iris.target
    X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = TEST_SIZE)
    return X_train,X_test, y_train, y_test, X, y


def run_logistic_regression(X_train,y_train,X_test):
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    pred = lr.predict(X_test)
    return pred

def return_shuffled_datasets(X,y,test_size):
    train_size = 1-test_size 
    length = len(X)
    shuf = np.random.permutation(length)
    ntrain_sz = int(train_size * (length))
    X_train_val = X[shuf[:ntrain_sz]]
    y_train_val = y[shuf[:ntrain_sz]]
    X_test_val = X[shuf[ntrain_sz:]]
    y_test_val = y[shuf[ntrain_sz:]]
    return X_train_val, X_test_val, y_train_val, y_test_val

# Evaluate performance of the two models side by side
def evaluate_errors(X,y,npermutation = 100):
    test_set = np.arange(0.1,0.5,0.1)  # Size of the test size we will consider, training size = 1 - test_size
    average_error_naive_bayes = []
    average_error_logistic_regression = []
    for TEST_SZ in test_set:
        errors = {'naive_bayes':0.0, 'logistic_regression':0.0}
        for i in range(npermutation):
            X_train_val, X_test_val, y_train_val, y_test_val = return_shuffled_datasets(X,y,TEST_SZ)
            # Fit our NB
            NB = NaiveBayes()
            NB.fit(X_train_val,y_train_val)
            predictions = NB.predict(X_test_val)
            errors['naive_bayes'] += (1.0 - np.mean(predictions == y_test_val))

            lr = LogisticRegression()
            lr.fit(X_train_val, y_train_val)
            predictions = lr.predict(X_test_val)

            errors['logistic_regression'] += (1.0- np.mean(predictions == y_test_val))

        average_error_naive_bayes.append(errors['naive_bayes']/npermutation)
        average_error_logistic_regression.append(errors['logistic_regression']/npermutation)
    return average_error_naive_bayes, average_error_logistic_regression, test_set


def plot_errors(err_nb, err_lr, train_set):
    plt.figure(figsize=(10,6))
    plt.title("Error plot - NB vs LR @ 200 random selections of training dataset")
    plt.xlabel("Training data size in % of the total data")
    plt.ylabel("Avg error over 200 permutations")
    plt.plot(train_set,err_nb,label = "Naive bayes error")
    plt.plot(train_set, err_lr, label = "Logistic regression error")
    plt.show()

if __name__ == "__main__":
    X_train,X_test, y_train, y_test,X,y = import_data()
    NB = NaiveBayes(alpha= 10)
    NB.fit(X_train, y_train)
    predictions = NB.predict(X_test)
    accuracy = np.average([predictions == y_test])
    # Now that we have got Naive bayes

    predictinos_from_logistic_regression = run_logistic_regression(X_train, y_train, X_test)
    accuracy_lr = np.average([predictinos_from_logistic_regression == y_test])

    average_error_naive_bayes,average_error_logistic_regression,test_set = evaluate_errors(X,y,100)
    train_set  = [1-x for x in test_set]
    plot_errors(average_error_naive_bayes, average_error_logistic_regression, train_set)

