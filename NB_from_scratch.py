import warnings
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
warnings.filterwarnings(action="ignore")


# Set seeds
import random 
random.seed(100)
np.random.seed(100)

TEST_SIZE  = 0.2


class NaiveBayes:

    @staticmethod
    def find_unique_classes_with_count(a):
        _class = np.asarray(np.unique(a,return_counts=True)).T
        return __class[:0], _class[:1]
    
    def __init__(self, alpha = 1):
        self.alpha = alpha

    # Calculate P(C1)
    def calculate_class_probabilities(self):

        for idx, cls in enumerate(self.classes):
            self.class_probabilities[cls] = self.class_freq[idx] /  self.length
    
    # Calculate P(X | C1)
    def calculate_likelihood_probabilities(self):
        
        for idx, cls in self.classes:
            indices = np.where(self.y_train == cls)
            # Get the filtered dataframe of our interest
            X_temp = self.X_train[indices]

            # Now, go feature by feature to calcu

    # PDF of the gaussian distribution to compute stats



    def calculate_statistics(self):
        self.mean = np.zeros((len(self.classes), self.X.shape[1])) 
        self.std = np.zeros((len(self.classes), self.X.shape[1]))
        for idx, cls in enumerate(self.classes):
            self.mean[idx:] = self.X[self.y == cls].mean(axis = 0)
            self.std[idx:] = self.X[self.y == cls].std(axis = 0)


    def fit(self, X_train, y_train):
        
        self.classes, self.class_freq = NaiveBayes.find_unique_classes_with_count(y_train)
        self.length = len(X_train)

        self.class_probabilities = {}
        # Calculating the class probabilities
        self.calculate_class_probabilities()
        
        # Calculating likelihood probabilities
        self.calculate_likelihood_probabilities()

        # Calculate the mean, std dev for Gaussian transform
        self.calculate_statistics()


    
    def _predict(self, x):
        "x: single instance of 1 row and d column size"
        pass
    
    def predict(self,X_test):
        preditions = []
        for x in X_test:
            predictions.append(self._predict(x))
        return predictions

    def __repr__(self):
        return "Class values are {}".format(self.classes) 


#Import the datasets as X,y and return as train and test in proportion to pre-set split ratio
def import_data():
    iris = load_iris()
    X,y = iris.data, iris.target
    X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = TEST_SIZE)
    return X_train,X_test, y_train, y_test

# Evaluate performance of the two models side by side
def evaluate_errors():
    pass

if __name__ == "__main__":
    X_train,X_test, y_train, y_test = import_data()
    NB = NaiveBayes(X_train, y_train)
    NB.fit()

