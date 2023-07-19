#!/usr/bin/env python
# coding: utf-8

# In[1]:


# general use
import os
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.model_selection import train_test_split

# for evaluation
from numpy import mean, std
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

# for current method
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# In[2]:


data_dirpath = 'dataset'
train_name = 'train.csv'
test_name = 'test.csv'

train_path = os.path.join(data_dirpath, train_name)
test_path = os.path.join(data_dirpath, test_name)
train_df = pd.read_csv(train_path, header=[0])
test_df = pd.read_csv(test_path, header=[0])

print(f'[Default] Number of train data: {train_df.shape[0]}, Number of test data: {test_df.shape[0]}')

train_df.head()


# In[3]:


# G1
train_df["Male/Female Actors Ratio"] = (train_df['Number of male actors']+1) /(train_df['Number of female actors']+1)
train_df["Log Male/Female Actors Ratio"] = np.log(train_df["Male/Female Actors Ratio"])
train_df["Total Actors"] = train_df['Number of male actors']+train_df['Number of female actors']
# train_df["LogGross"] = np.log(train_df["Gross"])
# train_df.drop(['Male/Female Actors Ratio', 'Number of male actors', 'Number of female actors', 'Total Actors'], axis=1, inplace=True)


# C2
train_df.loc[train_df['Lead'] == 'Male', 'Total Male Words'] = train_df['Number words male'] + train_df['Number of words lead']
train_df.loc[train_df['Lead'] != 'Male', 'Total Male Words'] = train_df['Total words'] - (train_df['Number words female'] + train_df['Number of words lead'])

train_df.loc[train_df['Lead'] == 'Female', 'Total Female Words'] = train_df['Number words female'] + train_df['Number of words lead']
train_df.loc[train_df['Lead'] != 'Female', 'Total Female Words'] = train_df['Total words'] - (train_df['Number words male'] + train_df['Number of words lead'])
# train_df["Total Male/Female Words Ratio"] = train_df["Total Male Words"] / train_df["Total Female Words"]
train_df.drop(['Number words female', 'Number words male'], axis=1, inplace=True)

# W1
train_df['Words per Male Actor'] = train_df['Total Male Words'] / train_df['Number of male actors']
train_df['Words per Female Actor'] = train_df['Total Female Words'] / train_df['Number of female actors']


# W2
train_df['Log Words per Male Actor'] = np.log(train_df['Words per Male Actor'])
train_df['Log Words per Female Actor'] = np.log(train_df['Words per Female Actor'])

train_df.drop(['Words per Male Actor', 'Words per Female Actor'], axis=1, inplace=True)
train_df.drop(['Male/Female Actors Ratio', 'Number of male actors', 'Number of female actors', 'Total Actors'], axis=1, inplace=True)


# train_df['Log Total Male Words'] = np.log(train_df['Total Male Words'])
# train_df['Log Total Female Words'] = np.log(train_df['Total Female Words'])
# train_df["Log Total Male/Female Words Ratio"] = np.log(train_df["Total Male/Female Words Ratio"])

# train_df['Lead Word Dominance'] = train_df['Number of words lead'] / train_df['Total words']

# train_df["Total Male Words Percentage"] = train_df["Total Male Words"]/train_df["Total words"]
# train_df["Total Female Words Percentage"] = train_df["Total Female Words"]/train_df["Total words"]
# train_df["Total Female Words Percentage"] = 1 - train_df["Total Male Words Percentage"]


# In[4]:


lead_map = {'Female': 0, 'Male': 1}
train_df['Lead'] = train_df['Lead'].map(lead_map).astype(int)

x_data=train_df.loc[:, train_df.columns != 'Lead']
y_data=train_df['Lead']

train_df.head()


# In[5]:


################################################
#                Test features                 #
################################################
# A
x_data["YearXGross"] = (x_data["Year"]) * x_data["Gross"]

# TW1
x_data["Other total words"] = x_data["Total words"] - x_data['Number of words lead']

# E1
# drop more data
x_data.drop(['Mean Age Male', 'Mean Age Female'], axis=1, inplace=True)

# E2
# drop more data
x_data.drop(['Total words'], axis=1, inplace=True)
x_data.drop(['Number of words lead'], axis=1, inplace=True)

# E3 decrease
# x_data.drop(['Number of male actors', 'Number of female actors'], axis=1, inplace=True)

# E4 
x_data.drop(['Age Lead', 'Age Co-Lead'], axis=1, inplace=True)

##################################################
#                Test Combinations
##################################################
# A+G1+C2+W1+W2+TW1+E1+E2+E4 
# depth=3 / learning_rate=0.2 / n_estimators=700
# Mean Accuracy: 0.913
# Mean Recall: 0.944
# Mean Precision: 0.940
# Mean F1: 0.941

# A+G1+C2+W2+TW1+E1+E2+E4 => W2 is better
# depth=3 / learning_rate=0.2 / n_estimators=700
# Mean Accuracy: 0.914
# Mean Recall: 0.944
# Mean Precision: 0.946
# Mean F1: 0.942

##################################################
##################################################

feature_names = x_data.columns.tolist()
x_data.head()


# In[6]:


# for cross vaidation
# X_train, X_test, y_train, y_test 
X_train1, X_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
X_train2, X_test2, y_train2, y_test2 = train_test_split(x_data, y_data, test_size=0.2, random_state=1)
X_train3, X_test3, y_train3, y_test3 = train_test_split(x_data, y_data, test_size=0.2, random_state=2)
X_train4, X_test4, y_train4, y_test4 = train_test_split(x_data, y_data, test_size=0.2, random_state=3)
X_train5, X_test5, y_train5, y_test5 = train_test_split(x_data, y_data, test_size=0.2, random_state=4)

X_trainset = (X_train1, X_train2, X_train3, X_train4, X_train5)
X_testset = (X_test1, X_test2, X_test3, X_test4, X_test5)
y_trainset = (y_train1, y_train2, y_train3, y_train4, y_train5)
y_testset = (y_test1, y_test2, y_test3, y_test4, y_test5)


# In[8]:


def adaboost_model(learning_rate, cross_valid, X_train, X_test, y_train, y_test, n_estimators, max_depth):
    model = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=max_depth),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm = 'SAMME.R'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    
    # get importance
    importance = model.feature_importances_
    
    
    return model, accuracy, recall, precision, f1, cm, importance


# In[9]:


cross_valid = 0
learning_rate = 0.2
n_estimators = 700
max_depths=[1, 2, 3, 4, 5]

# Now create a figure
sub_row = 6
sub_col = 1

# test the whole set
for idx, max_depth in enumerate(max_depths):

    cross_valid += 1
    model_list = []
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    cm_list = []
    imp_list = []
    
    for idx_dataset, (X_train, X_test, y_train, y_test) in enumerate(zip(X_trainset, X_testset, y_trainset, y_testset)):
        model, acc, recall, precision, f1, cm, imp = adaboost_model(learning_rate, cross_valid, X_train, X_test, y_train, y_test, n_estimators, max_depth)
        model_list.append(model)
        accuracy_list.append(acc)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        cm_list.append(cm) 
        imp_list.append(imp)
        

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

        # report performance
        print('Cross Vaidation Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    imp_list = np.array(imp_list)
    imp_avg = imp_list.mean(axis=0)

    print(f'depth={max_depth} / learning_rate={learning_rate} / n_estimators={n_estimators}')
    print(f'Mean Accuracy: {mean(accuracy_list):.3f}')
    print(f'Mean Recall: {mean(recall_list):.3f}')
    print(f'Mean Precision: {mean(precision):.3f}')
    print(f'Mean F1: {mean(f1_list):.3f}')
    print('=======================================')


# In[10]:


i = 1
clf = model_list[i]
cm = cm_list[i]


# In[11]:


group_names = ['TN', 'FP', 'FN','TP']
group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
plt.show()


# In[ ]:




