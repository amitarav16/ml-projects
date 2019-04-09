import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#importing dataset
data = pd.read_csv("irisdata.csv",header = None,names=["sepal length","sepal width","petal length","petal width","class"])
x = data.iloc[:,:4].values
y = data.iloc[:,4].values
data.head()
data.iloc[:,3].size
#visualizing
sns.boxplot(data = data,width =0.5,fliersize = 5)
# categorical_values to numerical
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
onehotencoder = OneHotEncoder(categorical_features = [4])
y = onehotencoder.fit_transform(y).toarray()
#creating training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
#creating model
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
y_pred
#measuring accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
