import pandas as pd
import numpy as np
import glob
import xlrd

data = pd.read_csv('Xtrain.csv', sep=',', header=None)
dataset = data.values
print dataset
