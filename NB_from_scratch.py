import warnings
import pandas as pd
import numpy as np

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


    # TODO: Fix the errors occuring here
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
    return X_train,X_test, y_train, y_test


def run_logistic_regression(X_train,y_train,X_test):
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    pred = lr.predict(X_test)
    return pred



# Evaluate performance of the two models side by side
def evaluate_errors(X,y,npermutation = 100):
    test_size = [np.arange(0.1,0.5,0.1)]  # Size of the test size we will consider, training size = 1 - test_size
    length = len(X_train)
    
    average_error_naive_bayes = []
    average_error_linear_regression = []
    

    for TEST_SZ in test_size:
        errors = {'naive_bayes':0, 'linear_regression':0}
        shuf = np.random.permutation(length)
        ntrain_sz = (1-TEST_SZ) * (length)
        X_train_val = X_train[shuf[:ntrain_sz]]
        y_train_val = y_train[shuf[:ntrain_sz]]
        X_test_val = X_test[shuf[ntrain_sz:]]
        y_test_val = y_test[shuf[ntrain_sz:]]
        for i in range(npermutation):
            X_train_val, y_train_val, X_test_val, y_test_val = return_shuffled_datasets(X_train,y)
            shuf = 


if __name__ == "__main__":
    X_train,X_test, y_train, y_test = import_data()
    NB = NaiveBayes(alpha= 10)
    NB.fit(X_train, y_train)
    predictions = NB.predict(X_test)
    accuracy = np.average([predictions == y_test])
    # Now that we have got Naive bayes

    predictinos_from_logistic_regression = run_logistic_regression(X_train, y_train, X_test)
    accuracy_lr = np.average([predictinos_from_logistic_regression == y_test])

    print(f"The accuracy of linear regression model is {accuracy_lr}")
