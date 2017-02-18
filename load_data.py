import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Normalizer
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

col = pd.get_dummies(X_train['color'])
col = col.T.reindex(['red','blue','green','black','white','yellow','other_color']).T.fillna(0)

race_ = pd.get_dummies(X_train['race'])

X_train = pd.concat([X_train, gender], axis = 1)
del X_train['gender']
X_train.columns = ['vj','age','exercise','competitive','height','weight', 'injury','color','sleep','race','female', 'male']

# compet
X_train['competitive'].replace(1, 2, inplace=True)
X_train['competitive'].replace(2, 1, inplace=True)
X_train['competitive'].replace(3, 0, inplace=True)
X_train['competitive'].replace(4, 1, inplace=True)


X_train = pd.concat([X_train, col], axis = 1)
X_train.columns = ['vj','age','exercise','competitive','height','weight', 'injury','color','sleep','race','female', 'male','red','blue','green','black','white','yellow','other_color']
del X_train['color']

# X_train = pd.concat([X_train, race_], axis = 1)
# del X_train['race']
# X_train.columns = ['vj','age','exercise', 'competitive', 'height','weight', 'injury','sleep','female', 'male', 'red','blue','green','black','white','yellow','other_color','w','b','ami','ai','c','k','j','bir','other_race']


# Replaced exercise values
X_train['exercise'].replace(1, 5, inplace=True)
X_train['exercise'].replace(2, 4, inplace=True)
X_train['exercise'].replace(4, 2, inplace=True)
X_train['exercise'].replace(5, 1, inplace=True)

# injury
X_train['injury'].replace(2, 0, inplace=True)
X_train['injury'].replace(3, 0, inplace=True)


X_train.to_csv('Xtrain.csv')
