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
        __class = np.asarray(np.unique(a,return_counts=True)).T
        print(__class[:,0])
        print(__class)
        return __class[:,0], __class[:,1]
    
    def __init__(self, alpha = 1):
        self.alpha = alpha

    # Calculate P(C1)
    def calculate_class_probabilities(self):

        for idx, cls in enumerate(self.classes):
            self.class_probabilities[cls] = self.class_freq[idx] /  self.length
    
    # PDF of the gaussian distribution to compute stats
    def gaussian_pdf(self,x,class_idx):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


    # TODO: Fix the errors occuring here
    def calculate_statistics(self):
        self.mean = np.zeros((len(self.classes), len(self.y_train)), dtype=np.float64)
        print(self.mean.shape)
        print() 
        self.var = np.zeros((len(self.classes), len(self.y_train)), dtype=np.float64) 
        for idx, cls in enumerate(self.classes):
                self.mean[idx,:] = self.X_train[self.y_train == cls].mean(axis = 0)
                self.var[idx,:] = self.X_train[self.y_train == cls].var(axis = 0)


    def fit(self, X_train, y_train):
        
        self.classes, self.class_freq = NaiveBayes.find_unique_classes_with_count(y_train)
        self.length = len(X_train)

        self.X_train = X_train 
        self.y_train = y_train

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
            likelihood = np.sum(np.log(NaiveBayes.gaussian_pdf(x,idx)))
            diff_cls_prediction.append(prior+likelihood)
        
        return self.classes[np.argmax(diff_cls_prediction)]
    
    def predict(self,X_test):
        predictions = []
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
    NB = NaiveBayes(alpha= 1)
    NB.fit(X_train, y_train)
    predictions = NB.predict(X_test)
    print("Here")