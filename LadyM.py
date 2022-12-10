#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:05:45 2022

@author: elva
"""

from selenium import webdriver
import datetime
import os
import pandas as pd
import re
import time
from bs4          import BeautifulSoup
from selenium     import webdriver
from selenium.webdriver.common.by import By

path_to_driver = "/Users/elva/Downloads/chromedriver"
driver         = webdriver.Chrome(executable_path=path_to_driver)
driver.get("http://www.google.com")
time.sleep(3)

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 5)
pd.set_option('display.max_colwidth', 10)
pd.set_option('display.width',800)
path  = "/Users/elva/Downloads/"
os.chdir(path)

driver = webdriver.Chrome()
# Creating the list of links.
links_to_scrape = ['https://www.yelp.com/biz/lady-m-cake-boutique-new-york-8?osq=Lady+M',
                   'https://www.yelp.com/biz/lady-m-cake-boutique-new-york-6?osq=Lady+M',
                   'https://www.yelp.com/biz/lady-m-cake-boutique-new-york-4?osq=Lady+M',
                   'https://www.yelp.com/biz/lady-m-cake-boutique-new-york-3?osq=Lady+M',
                   'https://www.yelp.com/biz/lady-m-cake-boutique-new-york-2?osq=Lady+M',
                   'https://www.yelp.com/biz/lady-m-cake-boutique-new-york?osq=Lady+M',
                   'https://www.yelp.com/biz/lady-m-cake-boutique-boston',
                   'https://www.yelp.com/biz/lady-m-cake-boutique-arlington-heights?osq=Lady+M',
                   'https://www.yelp.com/biz/lady-m-cake-boutique-los-altos?osq=Lady+M',
                   'https://www.yelp.com/biz/lady-m-cake-boutique-arcadia-4?osq=Lady+M',
                   'https://www.yelp.com/biz/lady-m-cake-boutique-los-angeles?osq=Lady+M',
                   'https://www.yelp.com/biz/lady-m-cake-boutique-irvine-3?osq=Lady+M',
                   'https://www.yelp.com/biz/lady-m-santa-clara?osq=Lady+M',
                   'https://www.yelp.com/biz/lady-m-mclean?osq=Lady+M']
l               = 0
one_link        = links_to_scrape[l]
driver.get(one_link)
time.sleep(3)

# Finding all the reviews in the website and bringing them to python
reviews_one_store = []

# Finding all the reviews in the website and bringing them to python
condition         = True

while (condition):

    reviews           = driver.find_elements(By.XPATH,"//div[@class=' review__09f24__oHr9V border-color--default__09f24__NPAKY']")
    r                 = 0


    for r in range(len(reviews)):
        one_review                   = {}
        one_review['scrapping_date'] = datetime.datetime.now()
        one_review['google_url']     = driver.current_url
        soup                         = BeautifulSoup(reviews[r].get_attribute('innerHTML'))
    
        try:
            one_review_text = soup.find('span', attrs={'class':'raw__09f24__T4Ezm'}).text
        except:
            one_review_text = ""
        one_review['one_review_text'] = one_review_text
        
        try:
            one_review_stars = re.findall('[0-9] [Ss]tar',reviews[r].get_attribute('innerHTML'))[0]
        except:
            one_review_stars = ""
        one_review['one_review_stars'] = one_review_stars
        reviews_one_store.append(one_review)
        
        try:
            one_review_date = soup.find('span', attrs={'class':'css-chan6m'}).text
        except:
            one_review_date = ""
        one_review['one_review_date'] = one_review_date
        
        try:
            one_review_UserOrigin = soup.find('div', attrs={'class':'responsive-hidden-small__09f24__qQFtj border-color--default__09f24__NPAKY'}).text
        except:
            one_review_UserOrigin = ""
        one_review['one_review_UserOrigin'] = one_review_UserOrigin
# This store does not have an address shown on Yelp, so we have to hand type the address        
#        store_address = "8718 West 3rd Street, Los Angeles, CA 90048"
#        one_review['one_review_store_address'] = store_address
        
        location = driver.find_elements(By.XPATH,"//span[@class=' raw__09f24__T4Ezm']")
        store_address = ""
        for i in range(0,3):
            store_address += location[i].text
            if i <= 1:
                store_address += ", "
        one_review['one_review_store_address'] = store_address
 
    # 
    try:
        # load the next page
        driver.find_elements(By.XPATH,"//span[@class='icon--24-chevron-right-v2 navigation-button-icon__09f24__Bmrde css-1kq79li']")[0].click()
        time.sleep(5)
    except:
        condition = False
        
len(reviews_one_store)
driver.close()

# Store scraped data into dataframes for each store
reviews_one_store = pd.DataFrame(reviews_one_store)

reviews_one_store.to_csv("LadyM_NY1.csv")
df_NY1 = reviews_one_store.copy()
df_NY1

reviews_one_store.to_csv("LadyM_NY2.csv")
df_NY2 = reviews_one_store.copy()
df_NY2

reviews_one_store.to_csv("LadyM_NY3.csv")
df_NY3 = reviews_one_store.copy()
df_NY3

reviews_one_store.to_csv("LadyM_NY4.csv")
df_NY4 = reviews_one_store.copy()
df_NY4

reviews_one_store.to_csv("LadyM_NY5.csv")
df_NY5 = reviews_one_store.copy()
df_NY5

reviews_one_store.to_csv("LadyM_NY6.csv")
df_NY6 = reviews_one_store.copy()
df_NY6

reviews_one_store.to_csv("LadyM_Boston.csv")
df_BOSTON = reviews_one_store.copy()
df_BOSTON

reviews_one_store.to_csv("LadyM_Chicago.csv")
df_CHICAGO = reviews_one_store.copy()
df_CHICAGO

reviews_one_store.to_csv("LadyM_CA1.csv")
df_CA1 = reviews_one_store.copy()
df_CA1

reviews_one_store.to_csv("LadyM_CA2.csv")
df_CA2 = reviews_one_store.copy()
df_CA2

reviews_one_store.to_csv("LadyM_CA3.csv")
df_CA3 = reviews_one_store.copy()
df_CA3

reviews_one_store.to_csv("LadyM_CA4.csv")
df_CA4 = reviews_one_store.copy()
df_CA4

reviews_one_store.to_csv("LadyM_CA5.csv")
df_CA5 = reviews_one_store.copy()
df_CA5

reviews_one_store.to_csv("LadyM_WashingtonDC.csv")
df_WashingtonDC = reviews_one_store.copy()
df_WashingtonDC

# Combining all the stores dataframe
All_LadyM_Stores = [df_NY1,df_NY2,df_NY3,df_NY4,df_NY5,df_NY6,df_BOSTON,df_CHICAGO,
              df_CA1,df_CA2,df_CA3,df_CA4,df_CA5,df_WashingtonDC]
All_LadyM_Stores_df = pd.concat(All_LadyM_Stores,ignore_index=True)
All_LadyM_Stores_df.to_csv("All_LadyM_Stores.csv")

# Checking null value, result indicating no missing value
All_LadyM_Stores_df.isnull().sum()

# Data cleaning, now we need to change the star rating into numeric numbers
df = All_LadyM_Stores_df.copy()

df['rating'] = df['one_review_stars'].str[0]

# Ploting frequency of rating graph
import matplotlib.pyplot as plt
x = df['rating']
plt.hist(x, bins= 'auto', color='#0504aa',
                            alpha=0.7, rwidth=0.9)
plt.grid(axis='y', alpha=0.5)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Frequency of Rating')

df['review_length'] = df['one_review_text'].apply(len)

# We filter out ratings that has 3 stars. So that we now only have good vs. bad ratings.
filter_df = df.loc[df['rating'] != '3']

# Here,  we use count vectorizer to get our x variables, words that has significant meaning 
# when predicting whether a review is good or bad.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

corpus = filter_df.one_review_text.to_list()
vectorizer = TfidfVectorizer()
vectorizer = CountVectorizer(
    max_df = 0.95,
    min_df = 20)
#    max_features = 400)
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
print(X.shape)

# We transform 4 and 5 star ratings to 1, which represents good review.
# We transform 1 and 2 star ratings to 0, which represents bad review.
# In the end, we get our targeted y variable, which now becomes binary.
#filter_df['rating'] = filter_df['rating'].map({'4': 1, '5': 1, '1': 0, '2': 0})
filter_df.loc[filter_df['rating'] == '1', 'rating'] = 0
filter_df.loc[filter_df['rating'] == '2', 'rating'] = 0
filter_df.loc[filter_df['rating'] == '4', 'rating'] = 1
filter_df.loc[filter_df['rating'] == '5', 'rating'] = 1
y = filter_df['rating']

# I will train variables machine learning models to fit the dataset.
# In the end, we will compare test score of different algorithms and find the best model.

# The best model that has the highest test score is xgboost.
# XG Boost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = XGBClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("XGBoost accuracy score on testing sample is: ",metrics.accuracy_score(y_test, y_pred))

# XGBoost accuracy score on testing sample is:  0.9196234612599565

y_score = clf.predict_proba(X_test)[:,1]
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_score)
print('roc_auc_score for XGBoost: ', roc_auc_score(y_test, y_score))

# roc_auc_score for XGBoost:  0.9552978529804658

# Create visualization for ROC Curve
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - XGBoost')
plt.plot(false_positive_rate, true_positive_rate,linewidth=3)
plt.plot([0, 1], ls="--",linewidth=3)
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

#Precision: 0.937
#Recall: 0.964
#Accuracy: 0.920
#F1 Score: 0.950

# Create visualization for confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)

group_names = ['True Negative','False Positive','False Negative','True Positive']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

ax.set_title('XGBoost Confusion Matrix\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1'])
ax.yaxis.set_ticklabels(['0','1'])

## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
import numpy as np

clf = XGBClassifier()
clf.fit(X, y)
shuffle = ShuffleSplit(n_splits=25, test_size=0.25)
CVInfo = cross_validate(clf, X, y, cv=shuffle,return_train_score=True)
print("XGBoost mean score on trainning sample is: ",np.mean(CVInfo['train_score']))
print("XGBoost mean score on testing sample is: ",np.mean(CVInfo['test_score']))

# XGBoost mean score on trainning sample is:  0.9979516908212561
# XGBoost mean score on testing sample is:  0.9176828385228094


# The second best model is logisitc regression.
# Logistic
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
import numpy as np

LogisticModel =  LogisticRegression(penalty="none",solver="lbfgs")
LogisticModel.fit(X, y)
shuffle = ShuffleSplit(n_splits=250, test_size=0.25)
CVInfo = cross_validate(LogisticModel, X, y, cv=shuffle,return_train_score=True)
print("Logistic mean score on trainning sample is: ",np.mean(CVInfo['train_score']))
print("Logistic mean score on testing sample is: ",np.mean(CVInfo['test_score']))

#Logistic mean score on trainning sample is:  1.0
#Logistic mean score on testing sample is:  0.9162230267921796

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
LogisticModel =  LogisticRegression(penalty="none",solver="lbfgs")
LogisticModel.fit(X_train, y_train)
y_score = LogisticModel.predict_proba(X_test)[:,1]
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_score)
print('roc_auc_score for Logistic: ', roc_auc_score(y_test, y_score))

# roc_auc_score for Logistic:  0.934973121922573

# Create visualization for ROC Curve
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Logistic')
plt.plot(false_positive_rate, true_positive_rate,linewidth=3)
plt.plot([0, 1], ls="--",linewidth=3)
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# The third model that fits the dataset well is knn.
# knn

# First, we run a test to find the optimal number of neighbors.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 21)
shuffle = ShuffleSplit(n_splits=25, test_size=0.25)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    CVInfo = cross_validate(knn, X, y, cv=shuffle,return_train_score=True)
    training_accuracy.append(np.mean(CVInfo['train_score']))
    test_accuracy.append(np.mean(CVInfo['test_score']))

fig, ax = plt.subplots()
ax.plot(neighbors_settings, training_accuracy, label="training accuracy")
ax.plot(neighbors_settings, test_accuracy, label="test accuracy")
ax.set_xlim(20, 0)
ax.set_ylabel("Accuracy")
ax.set_xlabel("n_neighbors")
ax.grid()
ax.legend()

# From the graph above, we can see that test score is the highest at number of neighbors = 6

from sklearn.metrics               import confusion_matrix, classification_report, precision_score
from sklearn                       import neighbors
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

X_trainValid, X_test, y_trainValid, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
knn = neighbors.KNeighborsClassifier(n_neighbors=6)
pred = knn.fit(X_train, y_train).predict(X_test)
# print(confusion_matrix(y_test, pred).T)
print(classification_report(y_test, pred, digits=3))

# f1-score accuracy: 0.855

knn = neighbors.KNeighborsClassifier(n_neighbors=6)
shuffle = ShuffleSplit(n_splits=200, test_size=.25)
CVInfo = cross_validate(knn, X, y, cv=shuffle,return_train_score=True,n_jobs=-1)
print("KNN mean score on trainning sample is", np.mean(CVInfo['train_score']))
print("KNN mean score on testing sample is", np.mean(CVInfo['test_score']))

# KNN mean score on trainning sample is 0.8598937198067632
# KNN mean score on testing sample is 0.8244279507603187

X_trainValid, X_test, y_trainValid, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
knn = neighbors.KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_score = knn.predict_proba(X_test)[:,1]
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_score)
print('roc_auc_score for KNN: ', roc_auc_score(y_test, y_score))

# roc_auc_score for KNN:  0.7615740292593525

# Create visualization for ROC Curve
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - KNN')
plt.plot(false_positive_rate, true_positive_rate,linewidth=3)
plt.plot([0, 1], ls="--",linewidth=3)
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# The following are models that I tried, but did not get a desired outcome. 
# The test scores of the 3 models are:
# Ridge: 0.31
# Lasso: 0.44
# Random Forest: 0.47

# Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)
ridge_cv = RidgeCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10]).fit(X_train, y_train)

#score
print("The train score for ridge model is {}".format(ridge_cv.score(X_train, y_train)))
print("The test score for ridge model is {}".format(ridge_cv.score(X_test, y_test)))

#The train score for ridge model is 0.727542287125857
#The test score for ridge model is 0.31362691440852486

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


fullModel = make_pipeline(Ridge())
#fullModel.fit(Xs,y)
param_grid={'ridge__alpha':[0.0001,0.001,0.01,0.05,0.1,0.25,0.5,1, 2,  5., 10., 100., 250., 500., 1000.]}
shuffle = ShuffleSplit(n_splits=100, test_size=0.25)
grid_search=GridSearchCV(fullModel,param_grid,cv=shuffle,
                              return_train_score=True,n_jobs=-1)
grid_search.fit(X, y)
results = pd.DataFrame(grid_search.cv_results_)
print(results[['rank_test_score','mean_test_score','param_ridge__alpha']])

print("best param:",grid_search.best_params_)
print("best model:",grid_search.best_estimator_)
print("best test score:",grid_search.best_score_)

# This is best model
best_model = grid_search.best_estimator_

#best param: {'ridge__alpha': 100.0}
#best model: Pipeline(steps=[('ridge', Ridge(alpha=100.0))])
#best test score: 0.4677234303976062

# Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)
lasso_cv = LassoCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10], random_state=0).fit(X_train, y_train)
print("The train score for lasso model is {}".format(lasso_cv.score(X_train, y_train)))
print("The test score for lasso model is {}".format(lasso_cv.score(X_test, y_test)))

#The train score for lasso model is 0.5783973119815845
#The test score for lasso model is 0.4409842327307425

# Random Foreset
from sklearn.model_selection import train_test_split
X_trainValid, X_test, y_trainValid, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Try for random forest and grid search
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
nmc = 100
#  Note:  no need for scaling, back to simple parameters
# set up dictionary for grid search
param_grid={'max_features':[25,50,75,100],'max_depth':[10,20,30,40,50],'n_estimators':[20]}
# set up cross-validation shuffles
cvf = ShuffleSplit(test_size=0.25,n_splits=nmc)
# set up search
grid_search=GridSearchCV(RandomForestRegressor(),param_grid,cv=cvf,return_train_score=True,n_jobs=-1)
# implement search
grid_search.fit(X_trainValid,y_trainValid)
# move results into DataFrame
results = pd.DataFrame(grid_search.cv_results_)
print(results[['rank_test_score','mean_test_score','param_max_features','param_max_depth']])

# Print best params and model
print("best param:",grid_search.best_params_)
print("best model:",grid_search.best_estimator_)
print("best test score:",grid_search.best_score_)

#best param: {'max_depth': 50, 'max_features': 100, 'n_estimators': 20}
#best model: RandomForestRegressor(max_depth=50, max_features=100, n_estimators=20)
#best test score: 0.47136428956902365


# Twitter API
import requests
import json

url = "https://twitter135.p.rapidapi.com/Search/"

querystring = {"q":"LadyM cake","count":"100"}

headers = {
	"X-RapidAPI-Key": "c277d26b03msh3d1f8e41c07ea98p1afda5jsn23956ca422d8",
	"X-RapidAPI-Host": "twitter135.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.json())

dictionary = response.json()

tweet_id_list = []
for key in dictionary['globalObjects']['tweets']:
    tweet_id_list.append(key)
print(tweet_id_list)

review = []
for item in tweet_id_list:
    review.append(dictionary['globalObjects']['tweets'][item]['full_text'])
print(review)

date = []
for item in tweet_id_list:
    date.append(dictionary['globalObjects']['tweets'][item]['created_at'])
print(date)

import pandas as pd

zipped = list(zip(date,review))
df_tweet = pd.DataFrame(zipped, columns=['Date', 'Tweet'])
print(df_tweet)

df_tweet_filter = df_tweet.copy()

df_tweet_filter = df_tweet_filter.drop(0)
df_tweet_filter = df_tweet_filter.drop(1)
df_tweet_filter = df_tweet_filter.drop(2)
df_tweet_filter = df_tweet_filter.drop(11)
df_tweet_filter = df_tweet_filter.drop(12)
df_tweet_filter = df_tweet_filter.drop(17)
print(df_tweet_filter)

df_tweet_filter['Date'] = pd.to_datetime(df_tweet_filter['Date']).dt.date
print(df_tweet_filter)

date2 = ['2017-04-02','2014-10-01','2017-01-17','2019-06-01','2014-04-12','2017-06-23','2016-05-05','2015-07-07',
        '2015-03-02','2014-04-01','2013-03-07','2014-10-26','2015-12-05','2018-07-10','2015-07-03','2015-05-03','2017-09-25',
        '2019-04-24','2017-09-23','2018-01-23','2017-02-15','2017-09-12','2011-10-29','2015-05-25','2018-12-28','2015-12-12',
        '2017-06-27','2014-05-01','2017-04-26','2015-09-09','2016-05-05','2018-06-29','2017-04-22','2011-05-10','2017-10-08',
        '2016-09-03','2011-09-24','2017-10-07','2016-10-12','2016-02-12','2016-04-19','2016-11-30','2013-10-25','2011-09-09',
        '2011-06-17','2016-12-27','2012-05-25','2015-06-25','2017-09-19','2013-07-21','2016-03-25','2015-09-01','2015-08-28',
        '2014-07-13','2014-07-13','2016-04-23','2013-10-18','2014-01-24','2018-04-19','2016-05-10']
tweet2 = ['Had dinner at @GyuKakuJBBQ #gyukujbbq #GyuKakythen of course have to save some ROOM for #desserts #ladym @ladymcakes#cake #ladymcakes',
         'Needed to come here for their delicious Signature #MilleCrepes before I fly back to SF #nyc #LadyM #cake',
         '💜 Fun Fact 💜 How to woo me.... 🥰 Besides all things #JK & @bts_twt, #LadyM ...."takes the cake" 😉👍🏼 I will forgive you for any wrongdoings. 😏 It is as simple as that 🤭 Now, finding a location is the issue 😆',
         'Needed to treat ourselves after a day of completing our checklist!#ladyM #cake #checkers #checkerscake #berryberrymist #nyceats',
         'momma got me a ladym cake last night. ☺️',
         'Today #blogcare was making the guest comment avatar a picture of my favorite #ladym crepe cake. Now all is right in the world.',
        'The Lady M Boutique in the building is finally open! Make it even harder to save money...😂😂 #ladym #cake',
         'Every single crêpe in a #LadyM cake is handmade by our chefs. This labor-intensive component is our signature.',
         'Celebrating @krystlemobayeni birthday with a hodgepodge of @ladymcakes',
         'Original Mille crepe at #LadyM. Cake with 20 layers of crepe. A slice of heaven!',
         'Yummy pastry assortment from #LadyM Cake Boutique @TheDoeFund @SweetNY party.',
         'That is how we eat the lovely #ladyM cake... :/ taste good as well: ) http://ift.tt/1uXX3fO',
         'Original & Green Tea Mille Crepes "...twenty lacy thin crêpes enveloping the lightest pastry cream..." #LadyM #cake',
         'All I want is a LadyM cake for my birthday. Please & thanks 😊',
         'Exhibit 1: Evidence : proof that the @Empirehk media team did, with some glee, attack my #ladym cake. #bodymissing',
         'We decided to treat the #birthday boy to different cakes today. #billysbakery #ladym #cake… http://ift.tt/1GJHez7',
         'Belated birthday cake is the best because it does not matter that you sort of do not deserve it anymore. @ladymcakes #LadyM #cake #nomnomnom',
         'Just ordered a #ladym cake for my birthday to be shipped all the way to GA. #worthit',
         'I can work for LadyM cake boutique as a trash can. Dude, they trash cakes like nothing... 😫',
        '#LadyM cake really lives up to the hype. It is so rich that it tried to talk me into voting republican while I was eating it',
          'Happy Valentines Day💋 #발렌타인데이 #크레페케이크 #비싸 #레이디엠케이크 #valentines #LadyM #Crepe #cake #thanks #love https://instagram.com/p/BQh0wq8gwMY/',
          'Totes digging into this ladym #cake right on the street like a total boss',
          'Mmmm u r sweet (@ LadyM Cake Boutique) http://4sq.com/rwgIz5',
          'Finally brought my friend to lady m bakery great time #ladym #cake',
          'Lady M getting me through this Friday matinee at phantombway  🍰 #ladym #cake #somuchcakebeforethreeoclock #fridaymatinee',
          'My ladyM cake left at Dons house is finally being retrieved!',
          'My moment today was #LadyM cake that is currently being served in my office #cake #TreatTuesday @ladymcakes #yum',
          'Sun is out. Time for some #afternoontea 😜🍵🍰#ladyM #cake #bakery #tea #coffee #nomnomnom #nyc @ Lady M http://instagram.com/p/nd5iT1Qt4m/',
          'Nice one!😋🍰#desserts #instafood #ladym #cake #hk #instagood #delicious #causewaybay @ Lady m… https://instagram.com/p/BTWkV3ihiV2/',
          'LadyM chocolate cake, BryantPark at dusk, laughing friends. Best ending to a BDay. #THNXfortheloveyouall https://instagram.com/p/7bec2Jja7R/',
          'Girls love them. #ladym #cake #central #hongkong @ Lady M https://instagram.com/p/BFBfgDdpbMR/',
          'Tea time at beautiful Cake Boutique:)#ladym #cake #millecrapes #loveit #newyork #うま #最高 #休憩 #summer #greentea @ Lady M Cake Boutique ',
          'The famous #LadyM cake at #gyukaku - layers of crepes stacked with green tea cream. Good… https://instagram.com/p/BTNdT8IjzBA/',
          '*LadyM cake boutique* best cakes from the upper east! Must get a lady M Milla Crepes http://yfrog.com/h3a35nlj',
          'At LadyM cake in Irvine with multiple yummy special flavors, quite expensive but it is worth… https://instagram.com/p/BaAOMo8glBB/',
          'Parents enjoying their #anniversary #dessert #ladym #millecrepe #cake #eeeeeats #nyc https://instagram.com/p/BJ4V6-qBojw/',
          'I am at LadyM Cake Boutique (41 East 78th Street, New York) ',
          'The beginning of the festivities... #ladym #cake #crepes #champagne #25 #nyc #cheers… https://instagram.com/p/BZ8s1P2Bz2v/',
          '12/10/2016 - Just fell in love with their Mille-Crepes Cake! #LadyM #cake #millecrepe #cream… https://instagram.com/p/BLd62rhg0CJ/',
          'Because sometimes a girl deserves a pick-me-up😋😋😋 #tgif #sweet #afternoon #ladym #cake… https://instagram.com/p/BBtpRLCOMFl/',
          'Tasting Lady M new strawberry and mimosa cake! #ladym #cake #strawberry #mimosa #hcfood @ 海港城… https://instagram.com/p/BEYIe5jCyXt/',
          'Decided to eat our #LadyM #cake over at #bryantPark The park was so chill. Ang weird no? Pag… https://instagram.com/p/BNbV_F6gETN/',
          '#LADYM cake in NY delicious but horrible service whateves lol http://instagram.com/p/f5-f4WSK3k/',
          'Excited to host #Shabbat dinner on my balcony tonight. Even more excited for the #LadyM cake for dessert.',
          'LADYM CAKE BOUTIQUE 41 East 78th Street, New York, NY 10021 (212) 452-2222 http://zagat.com/node/3502202 Congratulations, LADYM!!!❤❤❤',
          'Lady M Marron cake was mine. Omg I am so in love #ladym #cake #marron #dessert https://instagram.com/p/BOif5LBD7aS/',
          'Spent the day with one of my favorite people @kitty8ee. Prada/Sciaperelli exhibit at   @ LadyM Cake Boutique http://instagr.am/p/LEDnYMI98u/',
          'Legendary Lady M cakes #ladym #cake #letscelebrate #mthlyparty #hcfood #harbourcity @ CMRS Digital… https://instagram.com/p/4WPIGrmk89/',
          '##matcha #millecrepes #gateau #cake from #ladym #cake #boutique a bit too much #cream devoured… https://instagram.com/p/BZPL2LVj2aJ/',
          'Lady M cake boutique now available at 40st bet 5-6 Ave. It is delicious like usual. What is your favorite? @ladyM #Cake #BryantPark #yummy',
          'Lady M treats after shopping 󾥢☕️󾌫 #LadyM #cake #coffee #Singapore http://fb.me/2UWPqG3BH',
          'Lady M. Not worth the hype. The strawberry shortcake was good though, so was the company 💕🍰 #ladym #cake https://instagram.com/p/7F7uDbnww4/',
          'Photo: #ladyM #cake festival with @tmwee21 @janesiewchien lol (at Lady M Singapore) http://tmblr.co/Z5TUTy1t1GUuN',
          'Yummy strawberry shortcake from LadyM cake boutique♡ http://instagram.com/p/qYci_fQjnB/',
          'Love the many layers of crepe and cream in a #ladym #cake - my fav green tea crepe cake!! #dessert… http://instagram.com/p/qYVdo2M0Rn/',
          '@ITCanadianGay 🍰 here. I would give u my LadyM cake lol but you far asf',
          'LadyM cake! It was soooo good!! ♡ @sherbetsheryl http://instagram.com/p/fmtPNMA-4X/',
          'A happy Comasgiving to us all! #ladym #cake #comasgiving bigchengtheory @devonmoran@lamp0687 jvaz71… http://instagram.com/p/jikIzMHkr5/',
          'When your here , I’ll make it all better with s slice of LadyM cake....or a bagel, you decide. In NYC, food is therapeutic.',
          'Green tea mille cake from #LadyM worst cake i have had in years... never again even if it is free.']

zipped2 = list(zip(date2,tweet2))
df_tweet2 = pd.DataFrame(zipped2, columns=['Date', 'Tweet'])
print(df_tweet2)

combined_df_tweet = pd.concat([df_tweet_filter, df_tweet2], ignore_index = True)
combined_df_tweet.reset_index()
print(combined_df_tweet)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

corpus = combined_df_tweet.Tweet.to_list()
vectorizer = TfidfVectorizer()
vectorizer = CountVectorizer(
    max_df = 0.95,
    min_df = 3)
X2 = vectorizer.fit_transform(corpus)
print(X2.shape)
vectorizer.get_feature_names()

y_pred2 = LogisticModel.predict_proba(X2)
y_pred2 = clf.predict_proba(X2)
print(y_pred2)


