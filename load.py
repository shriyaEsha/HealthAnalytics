import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, LabelEncoder, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
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

#Importing training data
X_train = pd.DataFrame()
df = []  
#X_train.columns = ['age','exercise','competitive','height','weight', 'gender','injury','color','sleep','race']
path = r'TestData'
filenames = glob.glob(path + "/*.xlsx")
for name in filenames:
	train = pd.read_excel(name, parse_cols='C', skiprows=[1,2], header=None)
	train = train.T
	df.append(train)

X_train = pd.concat(df)
X_train.columns = ['vj','age','exercise','competitive','height','weight', 'gender','injury','color','sleep','race']

# Changed gender from categorical to dummy
gender = pd.get_dummies(X_train['gender'])
X_train = pd.concat([X_train, gender], axis = 1)
del X_train['gender']

''' remove age '''
del X_train['age']

# X_train.columns = ['vj','age','exercise','competitive','height','weight', 'injury','color','sleep','race','female', 'male']
X_train.columns = ['vj', 'exercise','competitive','height','weight', 'injury','color','sleep','race','female', 'male']

# change injury
for i in xrange(X_train.shape[0]):
	v = X_train.iloc[i,5]
	if v == 1:
		X_train.iloc[i,5] = -1
	elif v == 2:
		X_train.iloc[i,5] = 1
	elif v == 3:
		X_train.iloc[i,5] = -1

# change exercise
for i in xrange(X_train.shape[0]):
	v = X_train.iloc[i,1]
	if v == 1 or v == 2:
		X_train.iloc[i,1] = 1
	elif v == 3:
		X_train.iloc[i,1] = 0
	else:
		X_train.iloc[i,1] = -1



''' try removing competitive completely '''
X_train['competitive'].replace(1, 5, inplace=True)
X_train['competitive'].replace(2, 1, inplace=True)
X_train['competitive'].replace(3, -1, inplace=True)
X_train['competitive'].replace(3, 0, inplace=True)

""" feature scale sleep """
# X_train.iloc[:,5] = preprocessing.normalize(X_train.iloc[:,5])
# X_train.iloc[:,6] = preprocessing.scale(X_train.iloc[:,6]).reshape(-1,1)

""" remove sleep """
del X_train['sleep']


# add competitive to X_train
# competitive_cats = ['competitive_agree', 'competitive_neither', 'competitive_disagree', 'competitive_none']
# Y = X_train['competitive']
# feature = Y.reshape(-1, 1)
# n = len(competitive_cats)+1
# onehot_encoder = OneHotEncoder(n_values=n,sparse=False, handle_unknown='ignore')
# a = onehot_encoder.fit_transform(feature).reshape((X_train.shape[0],n))
# a = np.delete(a, [0],1)
# for i in xrange(len(competitive_cats)):
# 	X_train[competitive_cats[i]] = a[:,i]
# del X_train['competitive']


''' try removing color completely '''
# add color to X_train
# color_cats = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'other']
# Y = X_train['color']
# feature = Y.reshape(-1, 1)
# n = len(color_cats)+1
# onehot_encoder = OneHotEncoder(n_values=n,sparse=False, handle_unknown='ignore')
# a = onehot_encoder.fit_transform(feature).reshape((X_train.shape[0],n))
# a = np.delete(a, [0],1)
# for i in xrange(len(color_cats)):
# 	X_train[color_cats[i]] = a[:,i]
del X_train['color']

# add color to X_train
X_train['race'].replace(1, 5, inplace=True)
X_train['race'].replace(2, 1, inplace=True)
X_train['race'].replace(3, -1, inplace=True)
X_train['race'].replace(3, 0, inplace=True)


# race_cats = ['white', 'black', 'ami', 'asin', 'chi', 'kor', 'jap','bir','other_race']
# Y = X_train['race']
# feature = Y.reshape(-1, 1)
# n = len(race_cats)+1
# onehot_encoder = OneHotEncoder(n_values=n,sparse=False, handle_unknown='ignore')
# a = onehot_encoder.fit_transform(feature).reshape((X_train.shape[0],n))
# a = np.delete(a, [0],1)
# for i in xrange(len(race_cats)):
# 	X_train[race_cats[i]] = a[:,i]
# del X_train['race']


''' deleting columns - color, competitive '''

print X_train



X_train.to_csv('Xtrain.csv', index=False)
# 13908.2585454 - without extra cols
# 104400.098433 - with extra cols
# 4749.57241234 - removing race as well
# 3567.63668519 - changing injury values
# 3282.79839935 - changing compet agree to 5 and replacing the others