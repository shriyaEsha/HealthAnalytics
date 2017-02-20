import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Normalizer, LabelEncoder, OneHotEncoder, PolynomialFeatures, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, accuracy_score
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import spatial
from sklearn import cross_validation
import math
from sklearn.svm import SVR
import glob
import xlrd
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer

def normalizeDataset(dataset):
    """
    :param dataset: data to be normalized
    :return: normalized data w.r.t the features
    """
    minMaxScaler = MinMaxScaler()
    xScaled = minMaxScaler.fit_transform(dataset)
    xNormalized = pd.DataFrame(xScaled)
    return xNormalized

def neuralNetworkRegression(X, Y, X_TEST, Y_TEST):
    """
    :param X: data consisting of features (excluding class variable)
    :param Y: column vector consisting of class variable
    :return: models neural network regression with fine-tuning of epochs
    """
    print "NEURAL NETWORK REGRESSION"
    print "Executing..."
    print

    # can change to model on the entire dataset but by convention splitting the dataset is a better option
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.10, random_state = 5)
    Y_test = Y_test.reshape(-1,1)
    Y_train = Y_train.reshape(-1,1)
    RMSEerror = []

    train = np.vstack((X_train, X_test))  # append both testing and training into one array
    outputTrain = np.vstack((Y_train, Y_test))
    outputTrain = [float(s.item()) for s in outputTrain]
    outputTrain = np.asarray(outputTrain, dtype=np.float64)
    # print outputTrain
    outputTrain = outputTrain.reshape( -1, 1 )

    inputSize = train.shape[1]
    targetSize = outputTrain.shape[1]

    ds = SupervisedDataSet(inputSize, targetSize)
    ds.setField('input', train)
    ds.setField('target', outputTrain)

    hiddenSize = 2
    epochs = 500  # got after parameter tuning

    # neural network training model
    net = buildNetwork( inputSize, hiddenSize, targetSize, hiddenclass=TanhLayer, bias = True )
    trainer = BackpropTrainer(net, ds)

    # uncomment out to plot epoch vs rmse
    # takes time to execute as gets best epoch value
    # getting the best value of epochs

    """
    print "training for {} epochs...".format( epochs )

    for i in range(epochs):
        print i
        mse = trainer.train()
        rmse = mse ** 0.5
        RMSEerror.append(rmse)

    plt.plot(range(epochs), RMSEerror)
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Epochs")
    plt.savefig("../Graphs/Network/Question 2c/RMSE vs Epochs.png")

    plt.show()
    """

    print "Model training in process..."
    train_mse, validation_mse = trainer.trainUntilConvergence(verbose = True, validationProportion = 0.15, maxEpochs = epochs, continueEpochs = 10)
    p = net.activateOnDataset(ds)

    mse = mean_squared_error(map(float, outputTrain), map(float, p))
    rmse = mse ** 0.5

    print "Root Mean Squared Error for Best Parameters : " + str(rmse)

    """ predict new value """
    prediction = net.activate(X_TEST)
    print "Predicted: ",prediction," True: ", Y_TEST#, "Error: ",np.sqrt(mean_squared_error(map(float,Y_test), ridge.predict(X_test)))


data = pd.read_csv('Xtrain.csv', sep=',', header=None)
dataset = data.values
header = dataset[0,1:dataset.shape[1]]
dataset = dataset[1:dataset.shape[0],:]
X = dataset[:,1:dataset.shape[1]]
y = dataset[:,0]
""" remove value from X for prediction """
X_TEST = X[0]
Y_TARGET = y[0]
X = X[1:]
y = y[1:]
def ridgeRegression(X,Y, X_test, Y_test):
    """
    :param X: data consisting of features (excluding class variable)
    :param Y: column vector consisting of class variable
    :return: report best RMSE value for tuned alpha in ridge regression
    """
    tuningAlpha = [0.1,0.01,0.001]

   # can change to model on the entire dataset but by convention splitting the dataset is a better option
   # X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.10, random_state = 5)

    ridge = RidgeCV(normalize=True,scoring='mean_squared_error', alphas=tuningAlpha, cv=10)
    ridge.fit(X, Y)
    prediction = ridge.predict(X)
    for (a,b) in zip(Y, prediction):
			print a," ",b

    print "RIDGE REGRESSION"
    print "Best Alpha value for Ridge Regression : " + str(ridge.alpha_)
    print 'Best RMSE for corresponding Alpha =', np.sqrt(mean_squared_error(Y, prediction))
    
    print "Predicting value for X_test: ", X_test
    print "Predicted: ",ridge.predict(X_test)," True: ", Y_test#, "Error: ",np.sqrt(mean_squared_error(map(float,Y_test), ridge.predict(X_test)))


# LASSO REGULARIZATION
def lassoRegularization(X,Y):
    """
    :param X: data consisting of features (excluding class variable)
    :param Y: column vector consisting of class variable
    :return: report best RMSE value for lasso regularization
    """
    tuningAlpha = [0.1,0.01,0.001]
    lasso = LassoCV(normalize=True, alphas=tuningAlpha, cv=10)
    lasso.fit(X,Y)
    prediction = lasso.predict(X)

    print
    print "LASSO REGULARIZATION"
    print "Best Alpha value for Lasso Regularization : " + str(lasso.alpha_)
    print 'Best RMSE for corresponding Alpha =', np.sqrt(mean_squared_error(Y, prediction))

def polynomialRegression(X,Y):
    """
    :param X: data consisting of features (excluding class variable)
    :param Y: column vector consisting of class variable
    :param datasetName: Network / Housing
    :return: model polynomial regression with tuning of degree of the features
    """
    print "POLYNOMIAL REGRESSION"
    print "Executing..."

    # can change to model on the entire dataset but by convention splitting the dataset is a better option
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.10, random_state = 5)

    # get the best degree of freedom
    degrees = range(1,10)
    rmse = []
    print "Testing for the best degree..."

    for deg in range(len(degrees)):
        polyFeatures = PolynomialFeatures(degree= degrees[deg],interaction_only=True, include_bias= False)
        lm = LinearRegression()

        pipeline = Pipeline([("polynomial_features", polyFeatures),
                             ("linear_regression", lm)])

        pipeline.fit(X,Y)
        scores = cross_validation.cross_val_score(pipeline, X, Y, scoring="mean_squared_error", cv=10)

        rmse.append(np.mean(abs(scores)**0.5))

    # plot graph for RMSE vs. degree of polynomial
    plt.figure(1)
    plt.plot(degrees,rmse)
    plt.ylabel("RMSE")
    plt.xlabel("Polynomial Degree")
    plt.title("RMSE vs Polynomial Degree")

    # best degree = housing - 3
    # best degree = network - 5

    degrees = [degrees[rmse.index(min(rmse))]]
    rmse = []

    # apply transformation on dataset to the best degree of freedom and then find error
    for deg in range(len(degrees)):
        polyFeatures = PolynomialFeatures(degree= degrees[deg],interaction_only=True, include_bias= False)
        X_train_trans = polyFeatures.fit_transform(X_train)
        X_test_trans = polyFeatures.fit_transform(X_test)

        lm = LinearRegression()
        lm.fit(X_train_trans,Y_train)
        predicted = lm.predict(X_test_trans)
        for (a,b) in zip(Y_train, predicted):
			print a," ",b

        rmse.append(np.mean(abs(predicted-map(float, Y_test))**0.5))
    print "Best Degree of Polynomial : " + str(degrees[0])
    print "Root Mean Squared Error for Best Parameters : " + str(rmse[0])

normalizedX = normalizeDataset(X)
print 'Normalized X: ', normalizedX
neuralNetworkRegression(normalizedX, y, X_TEST, Y_TARGET)
raw_input()
ridgeRegression(X, y, X_TEST, Y_TARGET)
lassoRegularization(X, y)
polynomialRegression(X, y)
raw_input()
