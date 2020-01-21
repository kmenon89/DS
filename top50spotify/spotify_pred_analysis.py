# Predictive Models for popularity
# Clustering visualization with insights for popularity and genres

import numpy as np
import pandas as pd
import pandas_profiling 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
#get data 

data_df=pd.read_csv(r'C:\backup\KAVITHA_BACKUP\Personal\DS_Kaggle\top50spotify\top50spotify2019\top50.csv', encoding = "ISO-8859-1")

data_df.drop(['Unnamed: 0'],axis=1,inplace=True)
data_df.head()

data_df.columns

data_df.describe()

#define predictors, target variable and input info

X=data_df[['Beats.Per.Minute', 'Energy','Danceability', 'Loudness..dB..', 'Liveness', 'Valence.', 'Length.',
       'Acousticness..', 'Speechiness.']]
y=data_df['Popularity']
info_df=data_df[['Track.Name', 'Artist.Name', 'Genre']]


#calculate the correlation between the features and target var using spearman correlation

sc = pd.concat([X,y],axis=1).corr(method='spearman')
sc.where()

# Generate a mask for the upper triangle
triangle_mask = np.zeros_like(sc, dtype=np.bool)
#triangle_mask
triangle_mask[np.triu_indices_from(triangle_mask)] = True

#plot the figure

plt.figure(figsize=(25,10))
sns.heatmap(data = sc, linewidths=.1, linecolor='black', vmin = -1, vmax = 1, mask = triangle_mask, annot = True,
            cbar_kws={"ticks":[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]})

plt.yticks(rotation=45)

#plt.show()

#no strong correlation found
#How is distributed average popularity over the genres


# Create aux dataframe for plot
df = pd.concat([info_df,y],axis=1)
# Plot
plt.figure(figsize = (25,10))
sns.barplot(data=df,x='Genre2',y='Popularity')
plt.xticks(rotation=45, ha='right')



# Even with most of the songs, Pop shows the lowest popularity average over all genres.
# Except for Pop, all average popularities are very close with Rap showing the higher
# I will discuss more this result after the next plot using the original genre distribution

# Plot
plt.figure(figsize = (25,10))
sns.barplot(data=df,x='Genre',y='Popularity')
plt.xticks(rotation=45, ha='right')

# Considering the previous two graphs, I believe that the low average popularity 
# of the Pop genre shown in the first one is due to the fact that the average metric penalizes 
# low values and some of the smaller genres pertaining to it such as australian pop, 
# canadian pop and boy band have the lowest values of popularity (presented in the last plot) 
# thus lowering the final average. Also, as the other genres created have much less music, 
# they end up benefiting (with less music, less chances to have a low popularity song).


#preicitve analysis Is it possible to predict Popularity based on X features
#using KNN,SVM,RANDOM FOREST
#as correlation is non linear and weak we willnot use linear models

#use cv 10 fold to get train and test set

#standardization

sc=StandardScaler()

#models 

svrclf=SVR()
rfclf=RandomForestRegressor()
knrclf=KNeighborsRegressor(n_neighbors=3)

metrics_svr=[]
metrics_rf=[]
metrics_knr=[]
#cross validation fold=10

cv=KFold(n_splits=10)
print(X)

def model_prep(clf,model_arr,X_train,X_test,y_train,y_test):
    #fit models
    clf.fit(X_train,y_train)
    #prediction
    y_pred_model=clf.predict(X_test)
    
    #calculate MSE and append to the array
    model_arr.append(mean_squared_error(y_test.values.ravel(),y_pred_model))


#loop to create the models
for train,test in cv.split(X):
    X_train,y_train,X_test,y_test=X.loc[train,:],y.loc[train],X.loc[test,:],y.loc[test]
    
    
    #fit and transform  train and test  set using standard scaler
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    
    #fit the models
    model_prep(svrclf,metrics_svr,X_train,X_test,y_train,y_test)
    model_prep(rfclf,metrics_rf,X_train,X_test,y_train,y_test)
    model_prep(knrclf,metrics_knr,X_train,X_test,y_train,y_test)
#print(metrics_svr)
np.mean(metrics_rf)
np.mean(metrics_svr)
np.mean(metrics_knr)


# print('RF average MSE: {1} \n  SVR average MSE: {2} \n KNR average MSE : {3}' 
# .format(np.mean(metrics_rf),np.mean(metrics_svr),np.mean(metrics_knr)))

#using feature selection ( Kbest) to select best features that improve model performace. 
#using loop to run up to 9 features

#standardization

sc=StandardScaler()

#models 

svrclf=SVR()
rfclf=RandomForestRegressor()
knrclf=KNeighborsRegressor(n_neighbors=3)

#var to get pred scores 
metricsRFR=[]
metricsSVR=[]
metricsKNR=[]
#var to keep mse values for various fs
metrics_svr_fs=[]
metrics_rf_fs=[]
metrics_knr_fs=[]
#cross validation fold=10

cv=KFold(n_splits=10)
#loop to create the models
for train,test in cv.split(X):
    X_train,y_train,X_test,y_test=X.loc[train,:],y.loc[train],X.loc[test,:],y.loc[test]

    #fit and transform using standard scalar

    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    #applying feature selection using Kbest looping to get all features
    for feat in range(1,10):
        #feature selection initiation
        #print(feat)

        kbfs=SelectKBest(score_func=f_regression,k=feat)
        X_train_fs=kbfs.fit_transform(X_train,y_train,center=True)
        X_test_fs=kbfs.transform(X_test)
        #fit models
        model_prep(svrclf,metrics_svr_fs,X_train_fs,X_test_fs,y_train,y_test)
        model_prep(rfclf,metrics_rf_fs,X_train_fs,X_test_fs,y_train,y_test)
        model_prep(knrclf,metrics_knr_fs,X_train_fs,X_test_fs,y_train,y_test)
    # Append Prediction Mean Square Error
    metricsRFR.append(metrics_rf_fs)
    metricsSVR.append(metrics_svr_fs)
    metricsKNR.append(metrics_knr_fs)
    # Reset our MSE lists
    metrics_svr_fs = []
    metrics_rf_fs = []
    metrics_knr_fs = []
metricsRFR
metricsSVR
metricsKNR

avgRFR = np.mean(metricsRFR, axis=0)
avgRFR

avgSVR = np.mean(metricsSVR, axis=0)
avgSVR

avgKNR = np.mean(metricsKNR, axis=0)
avgKNR

#conclusion
# Models failed to get a good MSE even after a feature selection because our target variable did 
# not change at all.
#Probably, this dataset is too small (only 50 samples) to get the desired answers 
