# read float data
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')
data2 = pd.read_csv('aasdf.csv')
data2.head()

x1=data2['Brake Pos sum'].values
x2=data2['Throttle Pos sum'].values
y1=data2['e'].values

plt.figure()
plt.plot(x1,y1)

plt.figure()
plt.plot(x2,y1)


# 1D data: quadratic fitting using numpy
import numpy as np
import matplotlib.pyplot as plt

ones = np.ones(len(x1))
xfeature = np.asarray(x1)
squaredfeature = xfeature ** 2
b = y1 # target value

# input preparation: [1 x x^2]
features = np.concatenate((np.vstack(ones),np.vstack(xfeature),np.vstack(squaredfeature)), axis = 1) 

param = np.linalg.lstsq(features, b)[0] # use least squares

plt.figure()
plt.scatter(xfeature,b)
u = np.linspace(int(np.min(xfeature)), int(np.max(xfeature)), 100) #np.linspace(0,len(x1),100)
plt.plot(u, u**2*param[2] + u*param[1] + param[0] )
plt.show()

# minimal solution
from scipy import optimize

def f1(x):# y=ax^2+bx+c
    return (x**2*param[2] + x*param[1] + param[0])

result = optimize.minimize_scalar(f1)
print(result)

x_opt=result.x # optimal solution
print(x_opt)

plt.plot(result.x,f1(result.x),'ro',markersize=10)
plt.xlabel("x")
plt.ylabel("$f_1(x)$")
plt.title("1D cost function")
plt.show()


# polynomical regression: using sklearn library
X = np.float64(x1).reshape(-1,1)#6 * np.random.rand(m, 1) - 3
y = np.float64(y1).reshape(-1,1)#0.5 * X**2 + X + 2 + np.random.randn(m, 1)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly_features.fit_transform(X)

X[0]
X_poly[0]

# linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_


##% Multiple linear regression
# Y=aX1+bX2+c

import pandas as pd
from sklearn import linear_model

# dictionary
dataSet={'Param1': x1,
         'Param2': x2,
         'LapTime':y1}
# tabular form
df = pd.DataFrame(dataSet,columns=['Param1','Param2','LapTime'])

X = df[['Param1','Param2']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['LapTime']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# prediction with sklearn
newParam1 = 3
newParam2 = 15
pred=regr.predict([[newParam1, newParam2]])
print ('Predicted Lap Time: \n', pred)

# optimize: full search
optX1=0
optX2=0
minV=10000

for i in np.linspace(np.min(x1),np.max(x1),100):
    for j in np.linspace(np.min(x2),np.max(x2),100):
        pred=regr.predict([[i, j]])
        print(pred)
        if pred < minV:
            optX1=i
            optX2=j
            minV=pred
            
            
print('Final solution:','x1=',optX1,'x2=',optX2)

