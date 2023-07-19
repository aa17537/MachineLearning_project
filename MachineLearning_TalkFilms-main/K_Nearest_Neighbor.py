#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports 
import os
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree
from sklearn.base      import clone
from sklearn.preprocessing import normalize

from sklearn.metrics import * 
from sklearn import metrics




# In[ ]:


# data paths
data_dirpath = 'dataset'
train_name = 'train.csv'
test_name = 'test.csv'

train_path = os.path.join(data_dirpath, train_name)
test_path = os.path.join(data_dirpath, test_name)
train_df = pd.read_csv(train_path, header=[0])
test_df = pd.read_csv(test_path, header=[0])
print(f'[Default] Number of train data: {train_df.shape[0]}, Number of test data: {test_df.shape[0]}')

#mreplacing male, female by zeor and one and separating the data from its label
lead_map = {'Female': 0 , 'Male': 1}
train_df['Lead'] = train_df['Lead'].map(lead_map).astype(int)

X_train = train_df.loc[:, train_df.columns != 'Lead']
y_train = train_df['Lead']


# ### Computing some more features for the data

# In[ ]:


#ration of males to females
X_train['female to male ratio'] = (X_train['Number words male']+1) /( X_train['Number words female']+1 )  

#the lead, colead Age difference
X_train['Age difference'] = X_train['Age Lead'] - X_train['Age Co-Lead']

#scaling the gross by the year in order to better account for inflation
X_train["YearXGross"] = (np.max(X_train["Year"].values))- X_train["Year"] * X_train["Gross"]

# words per actor for each gender:
X_train["words/actors female"] = X_train["Number words female"] / X_train["Number of female actors"]
X_train["words/actors male"] = X_train["Number words male"] / X_train["Number of male actors"]

# ratio of males to females, +1 because of division by zero
X_train["Male/Female Actors Ratio"] = (X_train['Number of male actors']+1) /(X_train['Number of female actors']+1)

# Total number of actors
X_train["Total Actors"] = X_train['Number of male actors']+X_train['Number of female actors']

X_train.head()


# ### scaling the data

# In[ ]:


# some features need to be grouped like all counts for words, otherwise their proportionality would be lost
combined_normalization = [
       ['Number words female','Number words male','Total words', 'Number of words lead','Difference in words lead and co-lead','words/actors female', 'words/actors male'],
       ['Number of male actors','Number of female actors', 'Total Actors'],
       ['Year'],
       ['Gross'],
       ['Mean Age Male', 'Mean Age Female', 'Age Lead', 'Age Co-Lead','Age difference'],
       ['female to male ratio'], 
       ['YearXGross'], 
       ['Male/Female Actors Ratio'],
 ]


for combined in combined_normalization:
    if len(combined) == 1:
        X_train[combined] = normalize(X_train[combined],axis = 0)
    else:
        X_train[combined] = normalize(X_train[combined],axis = 1)
                           
X_train.head()


# ### setting up KNN and parameters

# In[ ]:


metrics = KDTree.valid_metrics

# KNN
knn = KNeighborsClassifier()
knn.algorithm = 'kd_tree'
knn.metric = metrics[0]
knn.weights ='uniform' #'distance'
knn.n_neighbors = 5
knn.n_jobs = multiprocessing.cpu_count()-1 # to enable multithreading


# 
# feature is a list of features to drop from the specified training set. 
def evaluate_feature( training_set,feature_list = [], score_to_return = 'accuracy', n_neighbors = [3,5,7,8,9,10,11] , metrics = ['euclidean',  'chebyshev', 'infinity'], neighbors_model = knn ):
    from sklearn.model_selection import StratifiedKFold
    
    # setting up the data for crossvalidation
    X_train_restricted = training_set.drop(columns=feature) # here features are dropped. [] if no features are dropped
    crossValSegs = StratifiedKFold(n_splits=5,random_state = 42,shuffle = True).split(X_train_restricted,y_train) # setting up crossvalidation
    crossValSegs = list(crossValSegs)
    
    #initializing scoring measures
    scores_max = {
        'accuracy' : 0,
        'roc_auc' : 0,
        'precision': 0,
        'recall' : 0,
        'f1' : 0
    }
    max_confusion = []
    max_model_String = ''
    scores_all = {}
    

    for metric in metrics: #[KDTree.valid_metrics]: include all metrics if you wish
        neighbors_model.metric = metric
        for n in   n_neighbors: 
            scores_n ={ 
            'accuracy' : [],
            'roc_auc' :  [],
            'f1'       : [],
            'precision': [],
            'recall' : [],
            'confusion' : []
            }
            
            neighbors_model.n_neighbors = n
            for train, test in crossValSegs:
                neighbors_model.fit(X_train_restricted.values[train],y_train.values[train])
                probabilities = neighbors_model.predict_proba(X_train_restricted.values[test] )
                y_pred = neighbors_model.predict(X_train_restricted.values[test])
               
                scores_n['accuracy'].append(accuracy_score(y_train.values[test], y_pred))
                scores_n['f1'].append( f1_score(y_train.values[test], y_pred))
                scores_n['roc_auc'].append(roc_auc_score(y_train.values[test], y_pred)) 
                scores_n['confusion'].append(confusion_matrix(y_train.values[test], y_pred))
                scores_n['precision'].append(precision_score(y_train.values[test], y_pred))
                scores_n['recall'].append(recall_score(y_train.values[test], y_pred))
                
            mean_score = np.mean(scores_n[score_to_return])
            mean_confusion = np.mean(scores_n['confusion'],0)
            current =  f'n = {n} metric = {metric}     Features dropped = {feature}   Mean {score_to_return} : {mean_score}, confusion = {mean_confusion}' 
            if mean_score > scores_max[score_to_return]: 
                scores_max[score_to_return] = mean_score
                max_model_String = current
                max_confusion  = mean_confusion
                scores_all = scores_n.copy()
               
    scores_dict = { 
            'accuracy' : np.mean(scores_all['accuracy']),
            'roc_auc' :  np.mean(scores_all['roc_auc']),
            'f1'       : np.mean(scores_all['f1']),
            'precision': np.mean(scores_all['precision']),
            'recall' : np.mean(scores_all['recall']),
            'confusion' : np.mean(scores_all['confusion'],0)
            }

    print(f'\n\nThe Best Model Is {max_model_String}')
    scores = np.mean(scores_n[score_to_return])
    
    
    
    
    
    return scores_max[score_to_return], max_model_String, scores_dict


# #### evaluate dropping different features
# 

# In[ ]:


#drop each feature iteratively and evaluate the change in accuracy
all_features =  X_train.columns.tolist()
feature_score = []
feature_strings = []
score_dicts = []
score = 'roc_auc' # 'accuracy' 'roc_auc' ,'f1','precision', 'recall' 
for feature in all_features:
    print(f'current Feature = {feature}')
    max_accuracy, max_string, score_dict = evaluate_feature( X_train,[feature], score_to_return = score, metrics = ['euclidean',  'chebyshev', 'infinity'], neighbors_model = knn ) #['euclidean',  'chebyshev', 'infinity']
     
    feature_score.append(max_accuracy)
    feature_strings.append(max_string)
    score_dicts.append(score_dict.copy())
    
    
    
normal_score, _ ,_ = evaluate_feature( X_train,[], score_to_return = score, metrics = ['euclidean',  'chebyshev', 'infinity'], neighbors_model = knn )
    


# #### plotting the way that dropping different features affects the score
# 

# In[ ]:




fig = plt.figure()

all_features =  X_train.columns.tolist()


bestModel = feature_strings[np.argmax(feature_score)]
bestScores = score_dicts[np.argmax(feature_score)]

print(f'Best performance:\n {bestModel}')
print(f'best scores for best Model {bestScores}')
plt.barh(all_features, ((np.array(feature_score)-normal_score)*1))
plt.xlabel(f'improvement of {score} score by dropping features')
plt.ylabel('dropped feature')

plt.savefig('Figures/featureImportance_' + score + '.png', dpi='figure', format='png', 
        bbox_inches='tight', pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None
       )
plt.show()

feature_score = np.array(feature_score)





# remove all features that decrease the accuracy if they are included
feature_usefull = (feature_score-normal_score) > 0 

features =  [feat for usefull,feat in zip(feature_usefull,all_features) if   usefull]
print(features)
max_accuracy, max_string, _  = evaluate_feature( X_train,features, score_to_return = score, metrics = ['euclidean',  'chebyshev', 'infinity'], neighbors_model = knn )
    
print('\n\n',max_accuracy, max_string)

