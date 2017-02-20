import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, LabelEncoder, OneHotEncoder, PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn import cross_validation
import math
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import KFold
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
import pickle
from pybrain.datasets.supervised import SupervisedDataSet as SDS
import utils

""" Normalizing Dataset for Neural Network """
def normalizeDataset(dataset):
    """
    :param dataset: data to be normalized
    :return: normalized data w.r.t the features
    """
    minMaxScaler = MinMaxScaler()
    xScaled = minMaxScaler.fit_transform(dataset)
    xNormalized = pd.DataFrame(xScaled)
    return xNormalized

""" Neural Network """
def neuralNetworkRegression(X_test):
    """
    :param X: data consisting of features (excluding class variable)
    :param Y: column vector consisting of class variable
    :return: models neural network regression with fine-tuning of epochs
    """
    print "NEURAL NETWORK REGRESSION"
    print "Executing..."
    print

    print "Loading saved model..."
    net = pickle.load(open("Models/neural.sav", 'rb'))
    # utils.neuralNetworkRegression()
    """ predict new value """
    y_test = np.zeros((X_test.shape[0],1))
    input_size = X_test.shape[1]
    target_size = y_test.shape[1]
    ds = SDS( input_size, target_size )
    ds.setField( 'input', X_test)
    ds.setField( 'target', y_test)
    prediction = net.activateOnDataset( ds )
    print prediction
    return prediction
    
""" Ridge """
def ridgeRegression(X_test):
    """
    :param X: data consisting of features (excluding class variable)
    :param Y: column vector consisting of class variable
    :return: report best RMSE value for tuned alpha in ridge regression
    """
    print "Loading saved model..."
    ridge = pickle.load(open("Models/ridge.sav", 'rb'))
    """ predict new value """
    print ridge.predict(X_test)
    return ridge.predict(X_test)

""" Lasso """
def lassoRegularization(X_test):
    """
    :param X: data consisting of features (excluding class variable)
    :param Y: column vector consisting of class variable
    :return: report best RMSE value for lasso regularization
    """
    print "Loading saved model..."
    lasso = pickle.load(open("Models/lasso.sav", 'rb'))
    """ predict new value """
    print lasso.predict(X_test)
    return lasso.predict(X_test)

def all_three_models(X_test, header):

    """ Splitting Training and Testing for Model Creation """
    lm = pickle.load(open("Models/linear_3.sav", 'rb'))
    pred1 = lm.predict(X_test)
    print "Predicted: ", lm.predict(X_test)
    print '*'*80

    """ Ridge """
    clf = pickle.load(open("Models/ridge_3.sav", 'rb'))
    pred2 = clf.predict(X_test)
    print "Predicted: ", clf.predict(X_test)
    print '*'*80

    """ Lasso """
    clf = pickle.load(open("Models/lasso_3.sav", 'rb'))
    pred3 = clf.predict(X_test)
    print "Predicted: ", clf.predict(X_test)
    print '*'*80
    return pred1, pred2, pred3



