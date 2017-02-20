import pandas as pd
import numpy as np
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

''' remove age, sleep, color '''
del X_train['age']
del X_train['sleep']
del X_train['color']

# X_train.columns = ['vj','age','exercise','competitive','height','weight', 'injury','color','sleep','race','female', 'male']
X_train.columns = ['vj', 'exercise','competitive','height','weight', 'injury','race','female', 'male']

# change injury
for i in xrange(X_train.shape[0]):
	v = X_train.iloc[i,5]
	if v == 1:
		X_train.iloc[i,5] = -1
	elif v == 2:
		X_train.iloc[i,5] = 1
	elif v == 3:
		X_train.iloc[i,5] = 1

# change exercise
for i in xrange(X_train.shape[0]):
	v = X_train.iloc[i,1]
	if v == 1 or v == 2:
		X_train.iloc[i,1] = 1
	elif v == 3:
		X_train.iloc[i,1] = 0
	else:
		X_train.iloc[i,1] = -1

''' replacing competitive values '''
# X_train['competitive'].replace(1, 3, inplace=True)
# X_train['competitive'].replace(2, 1, inplace=True)
# X_train['competitive'].replace(3, -1, inplace=True)
# X_train['competitive'].replace(3, 0, inplace=True)

for i in xrange(X_train.shape[0]):
	v = X_train.iloc[i,2]
	if v == 1:
		X_train.iloc[i,2] = 5
	elif v == 2:
		X_train.iloc[i,2] = 4
	elif v == 4:
		X_train.iloc[i,2] = -2
	elif v == 5:
		X_train.iloc[i,2] = 1

'''exercise '''
for i in xrange(X_train.shape[0]):
	v = X_train.iloc[i,1]
	if v == 1:
		X_train.iloc[i,1] = 5
	elif v == 2:
		X_train.iloc[i,1] = 4
	elif v == 4:
		X_train.iloc[i,1] = -2
	elif v == 5:
		X_train.iloc[i,1] = 1

""" feature scale sleep """
# X_train.iloc[:,5] = preprocessing.normalize(X_train.iloc[:,5])
# X_train.iloc[:,6] = preprocessing.scale(X_train.iloc[:,6]).reshape(-1,1)


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


# del X_train['exercise']
del X_train['competitive']
# del X_train['injury']
# del X_train['race']
# del X_train['male']
# del X_train['female']
del X_train['height']
# del X_train['weight']


X_train.to_csv('Xtrain.csv', index=False)
# 13908.2585454 - without extra cols
# 104400.098433 - with extra cols
# 4749.57241234 - removing race as well
# 3567.63668519 - changing injury values
# 3282.79839935 - changing compet agree to 5 and replacing the others
"""Notes
1. Removed everything except male/female - 2504.98331066
2. Added hw - 2423.79388454
3. Added race - 2345.64448756
4. Removed race added injury - 2208.39160305
5. Added race and injury - 1939.75646558
6. Added exercise - 2441.6978294
7. Removed ex added compet - 2482.20321461
8. Added ex and compet - 2346.40388431
9. Changed params of compet from 5 to 3 - 2099.54020139
10. Added sleep - contributes vvvvvless
 feature    estim coef
0  height  5.807228e+00
1  weight -4.977832e-01
2  injury -4.171425e+01
3   sleep -7.105427e-15
4    race  1.354550e+01
5  female -3.997704e+01
6    male  3.997704e+01

"""