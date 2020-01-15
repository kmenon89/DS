#This dataset contains daily weather observations from numerous Australian weather stations.
#The target variable RainTomorrow means: Did it rain the next day? Yes or No.
#Note: You should exclude the variable Risk-MM when training a binary classification model.
#  Not excluding it will leak the answers to your model and reduce its predictability. Read more about it here.

import numpy as np 
import pandas as pd 

#importing the data from csv to the dataframe using pandas

df_weather=pd.read_csv(r'C:\backup\KAVITHA_BACKUP\Personal\DS_Kaggle\DS1\weather-dataset-rattle-package\weatherAUS.csv')

#check data distribution in each column 

df_weather.describe()

#check which columns have less data that can be avoided 

df_weather.count().sort_values()

#remove columns with less than 50 % values  and remove location and date as it is irrelevant for the data given
# also need to remove RISK_MM as it might leak information to the model. 

delete=["Sunshine","Evaporation","Cloud3pm","Cloud9am","Date","Location","RISK_MM"]

df_weather1=df_weather.drop(delete,axis=1)



#drop where values are nan
df_weather1.describe()
df_weather1.dropna(inplace=True)
df_weather1.describe()

# check if there are any outliers and see if they need to be removed
# df_weather1._get_numeric_data()

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x=df_weather1._get_numeric_data())

#plt.show()


from scipy import stats

z=np.abs(stats.zscore(df_weather1._get_numeric_data()))

df_w=df_weather1[(z<3).all(axis=1)]



# convert categorical columns in to numerical equivalent

cat_cols=["RainToday","RainTomorrow"]
for i in cat_cols:
    df_w[i].replace({'No':0,'Yes':1},inplace=True)


categorical_cols=["WindGustDir","WindDir9am","WindDir3pm"]
for i in categorical_cols:
    print(np.unique(df_w[i]))

df_w=pd.get_dummies(df_w,columns=categorical_cols)

df_w.head()

#let's standardize data:

from sklearn  import preprocessing

minmaxclf=preprocessing.MinMaxScaler()
df_scaled=pd.DataFrame(minmaxclf.fit_transform(df_w),index=df_w.index, columns=df_w.columns)#sets index and columns as per original df

df_scaled.head()

#feature selection  using select k best

#first we need to get x nd y

X=df_scaled.loc[:,df_scaled.columns != "RainTomorrow"]

y=df_scaled[['RainTomorrow']]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

feat_sel=SelectKBest(score_func=chi2,k=3)
X_new=feat_sel.fit_transform(X,y)

X_new
X.columns
X.columns[feat_sel.get_support(indices=True)]# why indices =true

#creating X using important features
X_f=df_scaled[['Rainfall', 'Humidity3pm', 'RainToday']]

#model selection
import time

def model_accuracy_calc(clf,model):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    X_train,y_train,X_test,y_test=train_test_split(X_f,y,test_size=0.25,random_state=0)
    #X_train,X_test,y_train,y_test=prereq()
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    score=accuracy_score(y_test,y_pred)
    print("accuracy for {0} : {1}".format(model,score))
    print("time taken:", time.time()-t0)


#1. logistic regression


from sklearn.linear_model import LogisticRegression

t0=time.time()
clf_lr=LogisticRegression(solver='liblinear')
# X_train,X_test,y_train,y_test=prereq()
# clf_lr.fit(X_train ,y_train)
# y_pred=clf_lr.predict(X_test)

# score_lr=accuracy_score(y_test,y_pred)

# print("accuracy for logistic regression:",score_lr)
# print("time taken:", time.time()-t0)
model_accuracy_calc(clf_lr,'logistic regression')

#2. Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

t0=time.time()
clf_rf=RandomForestClassifier(n_estimators=100,max_depth=5)
# X_train,X_test,y_train,y_test=prereq()
# clf_rf.fit(X_train,y_train)
# y_pred=clf_rf.predict(X_test)

# score_rf=accuracy_score(y_test,y_pred)

# print("accuracy for random forest :",score_rf)
# print("time taken:", time.time()-t0)
model_accuracy_calc(clf_rf,'random forest')

#3 Decision tree classifier

from sklearn.tree import DecisionTreeClassifier

t0=time.time()
clf_dt=DecisionTreeClassifier()
# X_train,X_test,y_train,y_test=prereq()
# clf_dt.fit(X_train,y_train)
# y_pred=clf_dt.predict(X_test)

# score_dt=accuracy_score(y_test,y_pred)

# print("accuracy for decison tree :",score_dt)
# print("time taken:", time.time()-t0)
model_accuracy_calc(clf_dt,'decison tree')

#4 SVM

from sklearn import svm

t0=time.time()
clf_svm=svm.SVC(kernel='linear')
# X_train,X_test,y_train,y_test=prereq()
# clf_svm.fit(X_train,y_train)
# y_pred=clf_svm.predict(X_test)

# score_svm=accuracy_score(y_test,y_pred)

# print("accuracy for SVC:",score_svm)
# print("time taken:", time.time()-t0)

model_accuracy_calc(clf_svm,'SVC')