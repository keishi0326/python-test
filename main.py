# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 07:50:21 2017

@author: kato
"""

import sys
import time,datetime
import pandas as pd
from sklearn import svm, grid_search, cross_validation, metrics
#from sklearn.ensemble         import RandomForestRegressor
from sklearn.ensemble         import GradientBoostingRegressor


start = time.time()

#学習データ
df = pd.read_csv('data/new_train.csv',sep=',', parse_dates = [0])
#df['datetime'] = df['datetime'].astype()

df['month'] = df['datetime'].dt.strftime('%m')
df['weekday'] = df['datetime'].dt.strftime('%w')
df['day'] = df['datetime'].dt.strftime('%d')


#テスト対象データ
df_t = pd.read_csv('data/test.csv',sep=',', parse_dates = [0])
#df['datetime'] = df['datetime'].astype()

df_t['month'] = df_t['datetime'].dt.strftime('%m')
df_t['weekday'] = df_t['datetime'].dt.strftime('%w')
df_t['day'] = df_t['datetime'].dt.strftime('%d')

df_date = df_t["datetime"].astype('str')

data_t = df_t.drop(["datetime"], axis=1)

print(df)

data = df.drop(["y","datetime"], axis=1)
target = df['y']

data_train_s, data_test_s, label_train_s, label_test_s = cross_validation.train_test_split(data, target, test_size=0.01)

#clf.fit(np.array(data_train_s), np.array(label_train_s))

#parameters = {
#        'n_estimators'      : [5],
#        'max_features'      : [3, 5, 7],
#        'random_state'      : [0],
#        'n_jobs'            : [4],
#        'min_samples_split' : [3],
#        'max_depth'         : [3]
#}

parameters = {
              'n_estimators' : [100, 500, 1000, 1500],                 
              'learning_rate' : [0.1, 0.05, 0.01, 0.005], 
              'max_depth': [4, 6, 8, 10],
              'min_samples_leaf': [3, 5, 9, 17, 20],
              'max_features': [1.0, 0.3, 0.1]
              }


clf_cv = grid_search.GridSearchCV(GradientBoostingRegressor(), parameters, cv = 4, scoring='neg_mean_absolute_error')

clf_cv.fit(data_train_s, label_train_s)

print("Best Model Parameter: ", clf_cv.best_params_)
print("Best Model Score: ", clf_cv.best_score_)

print("# PREDICT..")
pre = clf_cv.predict(data_test_s)

#ac_score = metrics.accuracy_score(label_test_s, pre)
ac_score = metrics.mean_absolute_error(label_test_s, pre)
print("正解率=", ac_score)


result = clf_cv.predict(data_t)

#df_date['result'] = result

df_result = pd.DataFrame()
df_result[1]=df_date
df_result[2]=result
         
now = datetime.datetime.now()      
filename = 'result' + now.strftime("%Y%m%d%H%m") + '.csv'         

df_result.to_csv(filename, sep=',', index=False, header=False)

elapsed_time = time.time() - start

print("elapsed_time:{0}".format(elapsed_time)+"[sec]")                        




