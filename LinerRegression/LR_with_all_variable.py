import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


'''There are different techniques to treat them, here I have used one hot
 encoding(convert each class of a categorical variable as a feature).
 Other than that I have also imputed the missing values for outlet size.'''

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

#imputing missing values
# print(train.isna().any())

train['Item_Weight'].fillna((train['Item_Weight'].mean()),inplace=True)
train['Item_Visibility'] = train['Item_Visibility'].replace(0,np.mean(train['Item_Visibility']))
train['Outlet_Establishment_Year'] = 2013 - train['Outlet_Establishment_Year']
train['Outlet_Size'].fillna('Small',inplace=True)

# creating dummy variables to convert categorical into numeric values
mylist = list(train.select_dtypes(include=['object']).columns)
dummies = pd.get_dummies(train[mylist],prefix=mylist)
train.drop(mylist,axis=1,inplace=True)
X= pd.concat([train,dummies],axis=1)

#Building the model

lreg = LinearRegression()

X = train.drop('Item_Outlet_Sales',1)
x_train,x_cv,y_train,y_cv = train_test_split(X,train.Item_Outlet_Sales,test_size=0.3)

#training a liner model
lreg.fit(x_train,y_train)

#predicting on cv
pred_cv = lreg.predict(x_cv)

#calculating mse
mse = np.mean((pred_cv - y_cv)**2)
print(mse)

#calculating R-square
r2 = lreg.score(x_cv,y_cv)
print(r2)


