import os
from bottle import route, run
import sys
import time,datetime
import pandas as pd
from sklearn import svm, grid_search, cross_validation, metrics
#from sklearn.ensemble         import RandomForestRegressor
from sklearn.ensemble         import GradientBoostingRegressor

 
@route("/")
def hello_world():
        return "Hello World!"
 
@route("/hello")
def hello_world():
#        return "hello"
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

        return time.time()
 
run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))