import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, LabelEncoder, OneHotEncoder, PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn import cross_validation
import math
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
import utils

if __name__ == '__main__':

    """ Reading data """
    data = pd.read_csv('Xtrain.csv', sep=',', header=None)
    dataset = data.values
    header = dataset[0,1:dataset.shape[1]]
    dataset = dataset[1:dataset.shape[0],:]
    X = dataset[:,1:dataset.shape[1]]
    y = dataset[:,0]
    """ remove value from X for prediction """
    X_TEST = X[5]
    Y_TARGET = y[5]
    # X = X[1:]
    # y = y[1:]

    """ Running Regression Algorithms """
    print "Running Neural"
    print "Normalizing data..."
    normalizedX = utils.normalizeDataset(X)
    neural = utils.neuralNetworkRegression(normalizedX, y, X_TEST, Y_TARGET)
    raw_input()
    
    print "Running Ridge CV..."
    ridge = utils.ridgeRegression(X, y, X_TEST, Y_TARGET)
    raw_input()

    print "Running Lasso CV"
    lasso = utils.lassoRegularization(X, y, X_TEST, Y_TARGET)
    raw_input()

    print "Running all three..."
    all_three = utils.all_three_models(X, y, header, X_TEST, Y_TARGET)
    print "Neural: ", neural," Ridge: ", ridge," Lasso: ",lasso, " Three: ", all_three

