import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Normalizer, LabelEncoder, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, accuracy_score
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import spatial
from sklearn.cross_validation import train_test_split
import math
from sklearn.svm import SVR
import glob
import xlrd
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = pd.read_csv('Xtrain.csv', sep=',', header=None)
dataset = data.values
header = dataset[0,1:dataset.shape[1]]
dataset = dataset[1:dataset.shape[0],:]
''' Split data into training and testing '''
X = dataset[:,1:dataset.shape[1]]
y = dataset[:,0]
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# fit model no training data
# model = XGBClassifier()
# model.fit(X_train, y_train)
# print(model)
# # make predictions for test data
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

lm = LinearRegression()
lm.fit(X_train, y_train)

# print coeff for each feature
coefs = pd.DataFrame(zip(header, lm.coef_), columns=['feature','estim coef'])
print coefs
print X_train[:,2], y_train

# scatter plot between exercise and vj - positive correlation
# plt.scatter(tuple(X_train[:,2]), y_train)
# plt.xlabel("Average freq of exercise")
# plt.ylabel("Average VJ")
# # plt.show()


# # scatter plot between gender-female and vj - positive correlation
# plt.scatter(tuple(X_train[:,3]), y_train)
# plt.xlabel("wt")
# plt.ylabel("Average VJ")
# plt.show()

""" scatter plot between weight and vj """
plt.scatter(tuple(X_train[:,3]), y_train)
plt.xlabel("wt")
plt.ylabel("Average VJ")
# plt.show()

''' prediction '''
# plt.scatter(tuple(y_test), lm.predict(X_test))
# plt.xlabel("VJ")
# plt.ylabel("Predicted VJ")
# plt.show()

'''rmse '''
target = map(float, y_test)
pred = lm.predict(X_test)
for (a,b) in zip(target, pred):
	print a," ",b
print np.mean((target - pred)**2)
# print mean_squared_error(target, pred)