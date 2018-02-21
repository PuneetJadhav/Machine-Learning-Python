import numpy as np
import warnings
import os
import pandas as pd
os.chdir("C:\\College\\Machine Learnings\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\Multiple_Linear_Regression")
warnings.filterwarnings('ignore')
startup = pd.read_csv("50_Startups.csv")
startup.head()
startup.tail()
startup.info()
### State has to be handeled for categorical variable
startup['State'].unique()
### There are 3 different states
startup[startup['R&D'].isna()]
### None missing values in R & D
startup[startup['Administration'].isna()]
#### No missing values for Administration
startup[startup['MarketingSpend'].isna()]
### No Missing values for marketing spend as well
startup[startup['State'].isna()]
### No Missing values for state
startup[startup['Profit'].isna()]
### No Missing values for Profit as well
X = startup.iloc[:,0:len(startup.columns)-1].values
Y = startup.iloc[:,len(startup.columns)-1:].values
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
warnings.filterwarnings('ignore')
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
std_X =StandardScaler()
std_Y=StandardScaler()
X_train =std_X.fit_transform(X_train)
X_test =std_X.transform(X_test)
Y_train=std_Y.fit_transform(Y_train)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

y_pred= regressor.predict(X_test)
np.round(std_Y.inverse_transform(y_pred),2)
def BackWardElimination(x,sl):
   
    for i in range(0,len(x[0])):
        regressor=sm.OLS(endog=Y,exog=x).fit()
        maxp=max(regressor.pvalues)
        if maxp > sl:
            for j in range(0,len(x[0])):
                if maxp == regressor.pvalues[j].astype('float'):
                    x=np.delete(arr=x,obj=j,axis=1)
    return x
    
               
sm.OLS(endog=Y,exog=X_opt).fit().summary()