# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:24:22 2018

@author: Pawankumar.Puthran
"""
#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import numpy as np

#----------------------------------------------------------------------------------------------------------------------------------------
#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 

#-----------------------------------------------------------------------------------------------------------------------------------------

import os
os.chdir('C:\\Users\\pawankumar.puthran\\Documents\\Miscellaneous\\Competition\\Data Challenge\\Data Challenge 2\\DataChallenge_22Jan2018\\')
import pandas as pd
trainData = pd.read_csv("credit_card_default_TRAIN.csv",skiprows=1)

#Renaming the columns
trainData.rename(columns={'PAY_0':'PAY_SEP',
    'PAY_2':'PAY_AUG',
    'PAY_3':'PAY_JUL',
    'PAY_4':'PAY_JUN',
    'PAY_5':'PAY_MAY',
    'PAY_6':'PAY_APR',
    'BILL_AMT1':'BILL_SEP',
    'BILL_AMT2':'BILL_AUG',
    'BILL_AMT3':'BILL_JUL',
    'BILL_AMT4':'BILL_JUN',
    'BILL_AMT5':'BILL_MAY',
    'BILL_AMT6':'BILL_APR',
    'PAY_AMT1':'PAY_AMT_SEP',
    'PAY_AMT2':'PAY_AMT_AUG',
    'PAY_AMT3':'PAY_AMT_JUL',
    'PAY_AMT4':'PAY_AMT_JUN',
    'PAY_AMT5':'PAY_AMT_MAY',
    'PAY_AMT6':'PAY_AMT_APR',
    },inplace=True)

#Creating new column for analysis - SPENT
trainData['SPENT_SEP'] = trainData.BILL_SEP - (trainData.BILL_AUG - trainData.PAY_AMT_SEP)
trainData['SPENT_AUG'] = trainData.BILL_AUG - (trainData.BILL_JUL - trainData.PAY_AMT_AUG)
trainData['SPENT_JUL'] = trainData.BILL_JUL - (trainData.BILL_JUN - trainData.PAY_AMT_JUL)
trainData['SPENT_JUN'] = trainData.BILL_JUN - (trainData.BILL_MAY - trainData.PAY_AMT_JUN)
trainData['SPENT_MAY'] = trainData.BILL_MAY - (trainData.BILL_APR - trainData.PAY_AMT_MAY) 

#Dropping rows with Negative spent
trainData.drop(trainData.loc[trainData['SPENT_SEP'] < 0].index,inplace=True)
trainData.drop(trainData.loc[trainData['SPENT_AUG'] < 0].index,inplace=True)
trainData.drop(trainData.loc[trainData['SPENT_JUL'] < 0].index,inplace=True)
trainData.drop(trainData.loc[trainData['SPENT_JUN'] < 0].index,inplace=True)
trainData.drop(trainData.loc[trainData['SPENT_MAY'] < 0].index,inplace=True) 

#Creating new column for analysis - %OFLIMIT_BAL
trainData['%LIMIT_BAL_SEP'] = (trainData.BILL_SEP / trainData.LIMIT_BAL)*100
trainData['%LIMIT_BAL_AUG'] = (trainData.BILL_AUG / trainData.LIMIT_BAL)*100
trainData['%LIMIT_BAL_JUL'] = (trainData.BILL_JUL / trainData.LIMIT_BAL)*100
trainData['%LIMIT_BAL_JUN'] = (trainData.BILL_JUN / trainData.LIMIT_BAL)*100
trainData['%LIMIT_BAL_MAY'] = (trainData.BILL_MAY / trainData.LIMIT_BAL)*100
trainData['%LIMIT_BAL_APR'] = (trainData.BILL_APR / trainData.LIMIT_BAL)*100

#Statistical information about the variables
trainData.describe()

"""
  ID       LIMIT_BAL           SEX     EDUCATION      MARRIAGE  \
count  22500.000000    22500.000000  22500.000000  22500.000000  22500.000000   
mean   11250.500000   163424.608000      1.610178      1.840667      1.564133   
std     6495.334864   128515.245979      0.487721      0.775181      0.521311   
min        1.000000    10000.000000      1.000000      0.000000      0.000000   
25%     5625.750000    50000.000000      1.000000      1.000000      1.000000   
50%    11250.500000   135000.000000      2.000000      2.000000      2.000000   
75%    16875.250000   230000.000000      2.000000      2.000000      2.000000   
max    22500.000000  1000000.000000      2.000000      6.000000      3.000000   

                AGE       PAY_SEP       PAY_AUG       PAY_JUL       PAY_JUN  \
count  22500.000000  22500.000000  22500.000000  22500.000000  22500.000000   
mean      35.212889      0.015067     -0.106978     -0.137422     -0.197333   
std        9.307266      1.119824      1.197125      1.202637      1.164344   
min       21.000000     -2.000000     -2.000000     -2.000000     -2.000000   
25%       28.000000     -1.000000     -1.000000     -1.000000     -1.000000   
50%       33.000000      0.000000      0.000000      0.000000      0.000000   
75%       41.000000      0.000000      0.000000      0.000000      0.000000   
max       79.000000      8.000000      8.000000      8.000000      8.000000   

                  BILL_JUN       BILL_MAY  \
count             22500.000000   22500.000000   
mean              42008.863511   39750.329956   
std               62189.619882   59596.457496   
min               70000.000000  -46627.000000   
25%               2400.000000    1795.750000   
50%               19051.000000   18259.500000   
75%               51814.500000   49635.500000   
max               891586.000000  927171.000000   

            BILL_APR    PAY_AMT_SEP   PAY_AMT_AUG    PAY_AMT_JUL  \
count   22500.000000   22500.000000  2.250000e+04   22500.000000   
mean    38353.361956    5495.471067  5.784070e+03    4870.560533   
std     58733.356897   15087.642904  2.113294e+04   15959.242382   
min   -339603.000000       0.000000  0.000000e+00       0.000000   
25%      1243.750000    1000.000000  7.980000e+02     367.000000   
50%     17175.000000    2098.000000  2.000000e+03    1676.000000   
75%     48739.750000    5000.000000  5.000000e+03    4193.750000   
max    961664.000000  505000.000000  1.684259e+06  896040.000000   

         PAY_AMT_JUN    PAY_AMT_MAY    PAY_AMT_APR  default payment next month  
count   22500.000000   22500.000000   22500.000000                22500.000000  
mean     4692.143200    4694.131200    5088.028222                    0.226133  
std     14823.164919   15023.608194   17300.349898                    0.418336  
min         0.000000       0.000000       0.000000                    0.000000  
25%       270.000000     247.000000      56.000000                    0.000000  
50%      1500.000000    1500.000000    1463.000000                    0.000000  
75%      4000.000000    4000.000000    4000.000000                    0.000000  
max    497000.000000  417990.000000  528666.000000                    1.000000  

           SPENT_AUG     SPENT_JUL      SPENT_JUN      SPENT_MAY  
count   22035.000000  2.203500e+04   22035.000000   22035.000000  
mean     8069.797096  8.880829e+03    6854.537009    6050.208668  
std     20477.481941  2.424445e+04   17477.504331   16511.078608  
min         0.000000  0.000000e+00       0.000000       0.000000  
25%       577.000000  4.900000e+02     393.000000     390.000000  
50%      2072.000000  1.884000e+03    1505.000000    1380.000000  
75%      6942.000000  7.500000e+03    5657.000000    4991.000000  
max    532286.000000  1.664163e+06  483713.000000  447736.000000 

"""

#Categorical values 
trainData['SEX'].value_counts()
'''
2    13729 - Female
1     8771 - Male
Name: SEX, dtype: int64
'''
trainData['EDUCATION'].value_counts()
'''
2    10634 - University
1     7982 - Graduate school
3     3581 - High school
5      184 - NA 
4       76 - Others
6       33 - NA
0       10 - NA
Name: EDUCATION, dtype: int64
'''
trainData['MARRIAGE'].value_counts()
'''
2    12219 - Single
1     9990 - Married
3      255 - Others
0       36 - NA
Name: MARRIAGE, dtype: int64
'''
trainData['AGE'].value_counts()
'''
29    1249
27    1158
28    1107
30    1042
26    1011
25     955
24     929
31     851
33     836
32     836
34     832
36     825
35     796
23     761
37     746
39     694
38     689
40     618
42     598
41     593
43     487
44     477
22     469
45     451
46     404
47     361
48     337
49     333
50     301
53     249
51     247
52     228
54     176
55     154
56     126
58      96
57      95
59      60
21      55
60      54
61      41
62      38
63      25
64      25
66      22
65      15
69      14
67      12
70       9
68       5
71       2
72       2
73       2
75       1
79       1
Name: AGE, dtype: int64
'''

#Distribution Analysis
trainData['PAY_APR'].hist(bins=50)
trainData['PAY_MAY'].hist(bins=50)
trainData['PAY_JUN'].hist(bins=50)
trainData['PAY_JUL'].hist(bins=50)
trainData['PAY_AUG'].hist(bins=50)
trainData['PAY_SEP'].hist(bins=50)

trainData['BILL_APR'].hist(bins=50)
trainData['BILL_MAY'].hist(bins=50)
trainData['BILL_JUN'].hist(bins=50)
trainData['BILL_JUL'].hist(bins=50)
trainData['BILL_AUG'].hist(bins=50)
trainData['BILL_SEP'].hist(bins=50)

trainData['PAY_AMT_APR'].hist(bins=50)
trainData['PAY_AMT_MAY'].hist(bins=50)
trainData['PAY_AMT_JUN'].hist(bins=50)
trainData['PAY_AMT_JUL'].hist(bins=50)
trainData['PAY_AMT_AUG'].hist(bins=50)
trainData['PAY_AMT_SEP'].hist(bins=50)


trainData.boxplot(column="default payment next month", by = "SEX")


#Category Variable Analysis
temp1 = trainData['default payment next month'].value_counts(ascending=True)
temp2 = trainData.pivot_table(values='SEX',index=['default payment next month'],aggfunc=lambda x: x.mean())
print("Frequency Table for Default Payment Next Month:") 
print(temp1)

print('\nProbility of Default Payment Next Month by SEX:') 
print(temp2)

'''
------------------------------------------------------------------ MODEL BUILDING --------------------------------------------------------
'''
#Logistic Regression
outcome_var = 'default payment next month'
model = LogisticRegression()
predictor_var = ['BILL_SEP','BILL_AUG','BILL_JUL','BILL_JUN','BILL_MAY','BILL_APR']#['AGE','SEX','EDUCATION','MARRIAGE','LIMIT_BAL']
classification_model(model, trainData,predictor_var,outcome_var)


#Decision Tree
model = DecisionTreeClassifier()
predictor_var = ['BILL_SEP','BILL_AUG','BILL_JUL','BILL_JUN','BILL_MAY','BILL_APR']
classification_model(model, trainData,predictor_var,outcome_var)


#Random Forest
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['PAY_SEP','PAY_AUG','PAY_JUL','PAY_MAY','BILL_SEP','BILL_AUG','BILL_JUL','BILL_JUN','BILL_MAY',
                 'BILL_APR','PAY_AMT_SEP','PAY_AMT_AUG','PAY_AMT_JUL','PAY_AMT_JUN','PAY_AMT_MAY','PAY_AMT_APR','AGE','EDUCATION',
                 'LIMIT_BAL']
classification_model(model, trainData,predictor_var,outcome_var)

#Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)
