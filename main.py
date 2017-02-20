import pandas as pd
import numpy as np
import math
import glob
import pandas as pd
import numpy as np
import math
import glob

#Importing training data
X_test = pd.DataFrame()
df = []  
path = r'TestData'
filenames = glob.glob(path + "/*.xlsx")
for name in filenames:
	train = pd.read_excel(name, parse_cols='C', skiprows=[1,2], header=None)
	train = train.T
	df.append(train)

X_test = pd.concat(df)
X_test.columns = ['vj','age','exercise','competitive','height','weight', 'gender','injury','color','sleep','race']

# Changed gender from categorical to dummy
gender = pd.get_dummies(X_test['gender'])
X_test = pd.concat([X_test, gender], axis = 1)
del X_test['gender']

''' remove age, sleep, color '''
del X_test['age']
del X_test['sleep']
del X_test['color']

# X_test.columns = ['vj','age','exercise','competitive','height','weight', 'injury','color','sleep','race','female', 'male']
X_test.columns = ['vj', 'exercise','competitive','height','weight', 'injury','race','female', 'male']

# change injury
for i in xrange(X_test.shape[0]):
	v = X_test.iloc[i,5]
	if v == 1:
		X_test.iloc[i,5] = -1
	elif v == 2:
		X_test.iloc[i,5] = 1
	elif v == 3:
		X_test.iloc[i,5] = 1

# change exercise
for i in xrange(X_test.shape[0]):
	v = X_test.iloc[i,1]
	if v == 1 or v == 2:
		X_test.iloc[i,1] = 1
	elif v == 3:
		X_test.iloc[i,1] = 0
	else:
		X_test.iloc[i,1] = -1

''' replacing competitive values '''
for i in xrange(X_test.shape[0]):
	v = X_test.iloc[i,2]
	if v == 1:
		X_test.iloc[i,2] = 5
	elif v == 2:
		X_test.iloc[i,2] = 4
	elif v == 4:
		X_test.iloc[i,2] = -2
	elif v == 5:
		X_test.iloc[i,2] = 1

'''exercise '''
for i in xrange(X_test.shape[0]):
	v = X_test.iloc[i,1]
	if v == 1:
		X_test.iloc[i,1] = 5
	elif v == 2:
		X_test.iloc[i,1] = 4
	elif v == 4:
		X_test.iloc[i,1] = -2
	elif v == 5:
		X_test.iloc[i,1] = 1


del X_test['exercise']
del X_test['competitive']
# del X_test['injury']
# del X_test['race']
# del X_test['male']
# del X_test['female']
del X_test['height']
# del X_test['weight']

X_test.to_csv('Xtest.csv', index=False)
