#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings
import pickle
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data_pd = pd.read_csv("dataset/diabetes.csv")
data = np.array(data_pd)

feature_names = data_pd.columns[:-1]
X = data_pd[feature_names]
y = data_pd.Outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


