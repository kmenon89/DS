# Predictive Models for popularity
# Clustering visualization with insights for popularity and genres

import numpy as np
import pandas as pd
import pandas_profiling 
import seaborn as sns
import matplotlib.pyplot as plt
#get data 

data_df=pd.read_csv(r'C:\backup\KAVITHA_BACKUP\Personal\DS_Kaggle\DS2\top50spotify2019\top50.csv', encoding = "ISO-8859-1")

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
triangle_mask
triangle_mask[np.triu_indices_from(triangle_mask)] = True

#plot the figure

plt.figure(figsize=(25,10))
sns.heatmap(data = sc, linewidths=.1, linecolor='black', vmin = -1, vmax = 1, mask = triangle_mask, annot = True,
            cbar_kws={"ticks":[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]})

plt.yticks(rotation=45)

plt.show()

#no strong correlation found
#How is distributed average popularity over the genres


# Create aux dataframe for plot
df = pd.concat([info,y],axis=1)
# Plot
plt.figure(figsize = (25,10));
sns.barplot(data=df,x='Genre2',y='Popularity');
plt.xticks(rotation=45, ha='right');



# Even with most of the songs, Pop shows the lowest popularity average over all genres.
# Except for Pop, all average popularities are very close with Rap showing the higher
# I will discuss more this result after the next plot using the original genre distribution

# Plot
plt.figure(figsize = (25,10));
sns.barplot(data=df,x='Genre',y='Popularity');
plt.xticks(rotation=45, ha='right');

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





