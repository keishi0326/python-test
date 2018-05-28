import sys, os, time, datetime

import pandas as pd
from sklearn import svm, grid_search, cross_validation, metrics
from sklearn.ensemble         import GradientBoostingRegressor

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

import common #original utility 
import boto3


df = pd.read_csv('data/promotion.csv',sep=',',encoding='SHIFT-JIS', parse_dates = [2,3])

df = common.transform(df)

print(df)

data = df.drop(["Time", "performance", "date"], axis=1)

#target = df['y']
target = df['performance']

data_train_s, data_test_s, label_train_s, label_test_s = cross_validation.train_test_split(data, target, test_size=0.01)
parameters = {
          'n_estimators' : [100, 500],                 
          'learning_rate' : [0.1], 
          'max_depth': [4],
          'min_samples_leaf': [9],
          'max_features': [1.0, 0.3]
          }

clf_cv = grid_search.GridSearchCV(GradientBoostingRegressor(), parameters, cv = 4, scoring='neg_mean_absolute_error')

clf_cv.fit(data_train_s, label_train_s)

print("Best Model Parameter: ", clf_cv.best_params_)
print("Best Model Score: ", clf_cv.best_score_)

file_name = "data/model_temp.pkl"

# 学習した分類器を保存する。
joblib.dump(clf_cv, file_name, compress=True)

print("Model save process normally end.")

S3_BUCKET = os.environ.get('S3_BUCKET')

file_type = "application/zip"

s3 = boto3.resource('s3')

# s3へのファイルアップロード
s3.meta.client.upload_file(file_name, S3_BUCKET, 'model.pkl')

print("Save the recommendation model on S3.")
