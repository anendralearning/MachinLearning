import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#reading files
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

#Column Item_weight has some Null values. SO below is the treatment for the same
train['Item_Weight'].fillna((train['Item_Weight'].mean()),inplace=True)
#importing Liner Regression sklearn

lreg = LinearRegression()

#splitting into training and cv for cross validation
# X = train.loc[:,['Outlet_Establishment_Year','Item_MRP']]
#if we include one more feature to predict the sale

X = train.loc[:,['Outlet_Establishment_Year','Item_MRP','Item_Weight']]
x_train,x_cv,y_train,y_cv = train_test_split(X,train.Item_Outlet_Sales)

#training the model
lreg.fit(x_train,y_train)

#predicting on cv
pred = lreg.predict(x_cv)

#calculating MSE
mse = np.mean((pred - y_cv)**2)

# print(mse)

#calculating coefficient
coeff = pd.DataFrame(x_train.columns)

coeff["coefficient_estimate"] = pd.Series(lreg.coef_)

print(coeff)


#calculating R-square

r2 = lreg.score(x_cv,y_cv)

print(r2)
