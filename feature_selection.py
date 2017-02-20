from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.cross_validation import train_test_split

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

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X_train, y_train)
# display the relative importance of each attribute
print(model.feature_importances_)