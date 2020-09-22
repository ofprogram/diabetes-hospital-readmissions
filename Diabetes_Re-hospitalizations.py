#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from scipy import interp
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Precision, Recall

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from pandas.plotting import scatter_matrix
from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.ensemble.partial_dependence import plot_partial_dependence

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, classification_report


# In[5]:


#Import data

def load():
    data = pd.read_csv(r'data/diabetic_data.csv')
    return data


# In[ ]:


#Pick the features of interest. 

def clean(data):
    df = data[['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses', 'A1Cresult', 'change', 'diabetesMed',
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
        'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']]

    #Change 'unknown' to 'other'

    df['race'] = df['race'].replace(['?'],'Other')

    #Define target variable. 

    data['readmitted'] = [ 0 if val == 'NO' else 1 for val in data['readmitted']]
    
    return df, data


# In[7]:


#Set X, y, and get dummy variables.

def get_dummies(df, data):

    X = pd.get_dummies(df, columns=['race', 'gender', 'age', 'admission_type_id', 
        'discharge_disposition_id', 'admission_source_id', 'A1Cresult', 'change', 
        'diabetesMed', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
        'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'])

    y = data['readmitted']
    
    return X, y


# In[ ]:


#Pie charts

df.race.value_counts().plot(kind='pie', figsize=(12,12), title='Race', fontsize=(15), style='fivethirtyeight')
df.gender.value_counts().plot(kind='pie', figsize=(12,12), title='Gender', fontsize=(15), style='fivethirtyeight')
df.age.value_counts().plot(kind='pie', figsize=(12,12), title='Age', fontsize=(15), style='fivethirtyeight')


# In[ ]:


#Bar Graphs

ave_meds = df[['age', 'num_medications']].groupby('age').mean().sort_values(by='age')
ave_procedures = df[['age', 'num_procedures']].groupby('age').mean().sort_values(by='age')
ave_lab_procedures = df[['age', 'num_lab_procedures']].groupby('age').mean().sort_values(by='age')
ave_time_spent = df[['age', 'time_in_hospital']].groupby('age').mean().sort_values(by='age')

ave_meds.num_medications.plot(kind='bar', title=' Average Number of Medications', figsize=(12,9), fontsize=(15), style='fivethirtyeight')
ave_lab_procedures.num_lab_procedures.plot(kind='bar', title='Average Number of Lab Procedures', figsize=(12,9), fontsize=(15), style='fivethirtyeight')
ave_procedures.num_procedures.plot(kind='bar', title='Average Number of Procedures', figsize=(12,9), fontsize=(15), style='fivethirtyeight')
ave_time_spent.time_in_hospital.plot(kind='bar', title='Average Time in Hospital', figsize=(12,9), fontsize=(15), style='fivethirtyeight')


# In[108]:


#Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)


# In[148]:


#Base ML Models

lr = LogisticRegression()
rf = RandomForestClassifier()
gdbc = GradientBoostingClassifier()

#Optimized ML Models

lr_1 = LogisticRegression(penalty='l2', tol=0.0001, C=0.01, fit_intercept=True, intercept_scaling=1.0)
rf_1= RandomForestClassifier(bootstrap=False, max_depth=None, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, n_estimators=80, random_state=1)
gdbc_1 = GradientBoostingClassifier(learning_rate=0.50, n_estimators=120, random_state=1)


# In[ ]:


#Classification Report

def class_report(model):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return classification_report(y_test, prediction)
  
#Base models

print(class_report(lr))
print(class_report(rf))
print(class_report(gdbc))

#Optimized models

print(class_report(lr_1))
print(class_report(rf_1))
print(class_report(gdbc_1))


# In[ ]:


#Feature importance graphs

def feat_importance(model): 
    model.fit(X_train, y_train)
    feat_scores = pd.DataFrame({'Fraction of Samples Affected' : model.feature_importances_}, index=X.columns)
    feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected')
    return feat_scores[135:].plot(kind='barh', title='Most Important Features', figsize=(12,9), fontsize=(15), style='fivethirtyeight')

print(feat_importance(rf))
print(feat_importance(gdbc))


# In[ ]:


#Neural Network

def basic_net(X_train, y_train):    
    n_feats = X_train.shape[1]

    model = Sequential() # sequence of layers

    hidden_units = 155
    n_classes = 2

    input_layer = Dense(units=hidden_units,
                    input_dim=n_feats,
                    kernel_initializer='constant',
                    activation='relu')

    hidden_layer = Dense(units=n_units,
                    kernel_initializer='constant',
                    activation='relu')

    output_layer = Dense(units=n_classes,
                    input_dim=hidden_units,
                    kernel_initializer='uniform',
                    activation='sigmoid')
    model.add(input_layer)
    model.add(hidden_layer)

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["f1score"])
    
    return model.fit(X_train[:100], to_categorical(y_train), epochs=10, batch=32)

print(model.summary())


# In[ ]:


#Random Forest Trees Vs. Accuracy Graph

def rf_chart(model):
    num_trees = range(5, 100, 5)
    accuracies = []
    for n in num_trees:
        tot = 0
        for i in range(5):
            model.fit(X_train, y_train)
            tot += rf.score(X_test, y_test)
        accuracies.append(tot / 5)
    fig, ax = plt.subplots()
    ax.plot(num_trees, accuracies)
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("Accuracy")
    ax.set_title('Accuracy vs Num Trees')


# In[ ]:


#Grid Search Random Forest

def grid_search_rf(X_train, y_train):
    random_forest_grid = {'max_depth': [25, 50, None],
                      'max_features': ['sqrt', 'log2', None],
                      'min_samples_split': [2, 4],
                      'min_samples_leaf': [1, 2, 4],
                      'bootstrap': [True, False],
                      'n_estimators': [10, 20, 40, 80, 100],
                      'random_state': [1]}

    rf_gridsearch = GridSearchCV(RandomForestClassifier(),
                             random_forest_grid,
                             n_jobs=-1,
                             verbose=True,
                             scoring='f1')

    rf_gridsearch.fit(X_train, y_train)

print("best parameters:", rf_gridsearch.best_params_)


# In[ ]:


#Grid Search Gradient Boost

def grid_search_gdbc(X_train, y_train):
    gradient_boost_grid = {'max_depth': [2, 4, 6, None],
                          'max_features': ['sqrt', 'log2', None],
                          'min_samples_split': [2, 4],
                          'min_samples_leaf': [1, 2, 4],
                          'n_estimators': [10, 20, 80, 100, 120],
                          'random_state': [1],
                          'learning_rate': [0.01, .1, .5, 1],
                          'subsample': [.25, .5, .75, 1]}


    gb_gridsearch = GridSearchCV(GradientBoostingClassifier(),
                                 gradient_boost_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 scoring='f1')
    gb_gridsearch.fit(X_train, y_train)

print("best parameters:", gb_gridsearch.best_params_)


# In[11]:


#ROC Curves

def plot_roc(X, y, clf_class, plot_name, **kwargs):
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)
    n_splits=5
    kf = KFold(n_splits=n_splits, shuffle=True)
    y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= n_splits
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random', figsize=(15,15))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:


plot_roc(X, y, rf, 'Random_Forest')
plot_roc(X, y, lr, 'Logistic_Regression')
plot_roc(X, y, gdbc, 'GradientBoosting')


# In[154]:


#Plot Confusion Matrix

def plot_conf_mat(model):

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    cm = [[tp,fp],[fn,tn]]

    plt.figure(figsize=(12,9))
    ax = sns.heatmap(cm, annot=True, fmt = "d", cmap="Spectral")

    ax.set_xlabel('ACTUAL LABELS')
    ax.set_ylabel('PREDICTED LABELS') 
    ax.set_title('Random Forest Confusion Matrix')
    ax.xaxis.set_ticklabels(['Yes', 'No'])
    ax.yaxis.set_ticklabels(['Yes', 'No'])
    plt.show()


# In[ ]:


print(plot_conf_mat(lr))
print(plot_conf_mat(rf))
print(plot_conf_mat(gdbc))


# In[ ]:





# In[ ]:




