import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import 


#importing the data file 
data_df=pd.read_csv(r'C:\backup\KAVITHA_BACKUP\Personal\Data science\Andrew NG\supervised\multivariate features\machine-learning-ex test week2\machine-learning-ex1\ex1\ex1data1.txt',header=None,names=['population','profit/loss'])
# Format with commas and round off to two decimal places in pandas
 
#pd.options.display.float_format = '{','}'.format
data_df.head()

#split features and predictor

X=data_df.iloc[:,0]
y=data_df.iloc[:,1]

#scatter plot 
plt.figure()
plt.scatter(X,y)
plt.xlabel('population in 10,000s')
plt.ylabel('profit in $10,000s')
plt.show()

#defining learning rate and sample size and range for running gradient descent

L=0.001 # learning rate alpha
theta0=0
theta1=0
epochs=1000

m=float(len(X))

# performing gradient descent

for i in range(epochs):
    y_pred=theta0*X+theta1
    Der_t0=(-2/m)*sum(X*(y-y_pred))
    Der_t1=(-2/m)*sum(y-y_pred)
    theta0=theta0-L*Der_t0
    theta1=theta1-L*Der_t1

print(theta0,theta1)

y_pred=theta0*X+theta1

#plot the linear regression line

plt.figure()
plt.scatter(X,y)
plt.plot([min(X),max(X)],[min(y_pred),max(y_pred)],color='red')
plt.show()


