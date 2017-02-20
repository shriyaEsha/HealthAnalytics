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

    hiddenSize = 1
    epochs = 800  # got after parameter tuning

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
    return prediction

""" Ridge """
def ridgeRegression(X,Y, X_test, Y_test):
    """
    :param X: data consisting of features (excluding class variable)
    :param Y: column vector consisting of class variable
    :return: report best RMSE value for tuned alpha in ridge regression
    """
    tuningAlpha = [0.1,0.01,0.001]

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
    return ridge.predict(X_test)

""" Lasso """
def lassoRegularization(X,Y, X_test, Y_test):
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
    print "Predicting value for X_test: ", X_test
    print "Predicted: ",lasso.predict(X_test)," True: ", Y_test#, "Error: ",np.sqrt(mean_squared_error(map(float,Y_test), ridge.predict(X_test)))
    return lasso.predict(X_test)

def all_three_models(X, y, header, X_TEST, Y_TARGET):

	""" Splitting Training and Testing for Model Creation """
	seed = 7
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=42)
    
        min_diff = 1000
        min_value = 1000
	print 'Linear Regression'
	lm = LinearRegression()
	lm.fit(X_train, y_train)
	# print coeff for each feature
	coefs = pd.DataFrame(zip(header, lm.coef_), columns=['feature','estim coef'])
	print coefs
	target = map(float, y_test)
	pred = lm.predict(X_test)
	for (a,b) in zip(target, pred):
		print a," ",b
	print"RMSE: ", np.sqrt(np.mean((target - pred)**2))
	print "Predicted: ", lm.predict(X_TEST)," True: ", Y_TARGET
	if abs(lm.predict(X_TEST) - float(Y_TARGET)) < min_diff:
		min_value, min_diff = lm.predict(X_TEST), abs(lm.predict(X_TEST) - float(Y_TARGET))
	print '*'*80

	""" Ridge """
	print "Ridge"
	clf = Ridge(alpha=1.0)
	clf.fit(X_train, y_train)
	coefs = pd.DataFrame(zip(header, clf.coef_), columns=['feature','estim coef'])
	print coefs
	target = map(float, y_test)
	pred = clf.predict(X_test)
	for (a,b) in zip(target, pred):
		print a," ",b
	print "RMSE: ", np.sqrt(np.mean((target - pred)**2))
	print "Predicted: ", clf.predict(X_TEST)," True: ", Y_TARGET
	if abs(clf.predict(X_TEST) - float(Y_TARGET)) < min_diff:
		min_value, min_diff = clf.predict(X_TEST), abs(clf.predict(X_TEST) - float(Y_TARGET))
	print '*'*80

	""" Lasso """
	print 'Lasso'
	clf = Lasso(alpha=1.0)
	clf.fit(X_train, y_train)
	coefs = pd.DataFrame(zip(header, clf.coef_), columns=['feature','estim coef'])
	print coefs
	target = map(float, y_test)
	pred = clf.predict(X_test)
	for (a,b) in zip(target, pred):
		print a," ",b
	print "RMSE: ", np.sqrt(np.mean((target - pred)**2))
	print "Predicted: ", clf.predict(X_TEST)," True: ", Y_TARGET
	if abs(clf.predict(X_TEST) - float(Y_TARGET)) < min_diff:
		min_value, min_diff = clf.predict(X_TEST), abs(clf.predict(X_TEST) - float(Y_TARGET))
	print '*'*80
	return min_value



