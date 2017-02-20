import pandas as pd
import numpy as np
import math
import glob

#Importing training data
X_train = pd.DataFrame()
df = []  
path = r'TrainData'
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


del X_train['exercise']
del X_train['competitive']
# del X_train['injury']
# del X_train['race']
# del X_train['male']
# del X_train['female']
del X_train['height']
# del X_train['weight']

X_train.to_csv('Xtrain.csv', index=False)
