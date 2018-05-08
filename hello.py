import os
from bottle import route, run
import sys
import time,datetime
import pandas as pd
from sklearn import svm, grid_search, cross_validation, metrics
#from sklearn.ensemble         import RandomForestRegressor
from sklearn.ensemble         import GradientBoostingRegressor
import psycopg2 #DB Connect

 
@route("/")
def hello_world():
        return "Hello World!"

@route("/db")
def hello_world():
	DATABASE_URL = os.environ['DATABASE_URL']

	conn = psycopg2.connect(DATABASE_URL, sslmode='require')
	
	print("DB connect successfull!")

	cur = conn.cursor()
	cur.execute("SELECT firstname, lastname, email FROM salesforce.contact")
	 
	for row in cur:
		print(row[0], row[1], row[2])
	
	return "select statement executed!"

@route("/dbupdate")
def hello_world():

	DATABASE_URL = os.environ['DATABASE_URL']
	
	try:
		conn = psycopg2.connect(DATABASE_URL, sslmode='require')
	
		print("DB connect successfull!")

		cur = conn.cursor()
	
		key = "00001_2"

		sql = """ UPDATE salesforce.CampaignCandidate__c SET order__c = order__c + 1
        	        WHERE foreignkey__c = %s"""

		#cur.execute("SELECT firstname, lastname, email FROM salesforce.contact")
	 
		cur.execute(sql, (key))
	
		conn.commit()
		cur.close()
	
		resultMsg = "Update statement executed!"
		
	except (Exception, psycopg2.DatabaseError) as error:
		resultMsg = "Database error occured."
		
	finally:
        	if conn is not None:
            		conn.close()

	return resultMsg

	
@route("/hello")
def hello_world():
#        return "hello")

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
#		return time.time()

		data = df.drop(["y","datetime"], axis=1)
		target = df['y']

		data_train_s, data_test_s, label_train_s, label_test_s = cross_validation.train_test_split(data, target, test_size=0.01)

		parameters = {
              'n_estimators' : [100],                 
              'learning_rate' : [0.1, 0.05], 
              'max_depth': [4, 6],
              'min_samples_leaf': [3, 9],
              'max_features': [1.0, 0.1]
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

		return "Predict process completed!"

run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
