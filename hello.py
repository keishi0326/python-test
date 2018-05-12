import os
from bottle import route, run, get, post, request
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

@route("/observation-result", method='GET')
def hello_world():
	empty = request.query.get('empty')
	storeID = request.query.get('storeID')

	#if input parameter is nothing, 'empty' literal set
	empty = "null!!" if empty is None else empty
	storeID = "null!!" if storeID is None else storeID	
	
	#return "Hello World!!!!!!!!!!!!"
	return '''empty param : {empty},  storeID param : {storeID}'''.format(empty=empty, storeID=storeID) 



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

@route("/dbinsert", method="GET")
def hello_world():

	id = request.query.get('key')

	id = id if id is not None else "CK-00000010"
	
	DATABASE_URL = os.environ['DATABASE_URL']
	
	try:
		conn = psycopg2.connect(DATABASE_URL, sslmode='require')
	
		print("DB connect successfull!")

		cur = conn.cursor()
	
		sql = """ INSERT INTO salesforce.CampaignCandidate__c(CampaignCandidateID__c, cm1_sfid__c, cm2_sfid__c, cm3_sfid__c, cm4_sfid__c, cm5_sfid__c, storesfid__c) VALUES (%s, %s, %s, %s, %s, %s, %s)"""

		key = ("CK-00000012", "a027F00000JWCIwQAP", "a027F00000JWCJQQA5", "a027F00000JWCJVQA5", "a027F00000JWCJaQAP", "a027F00000JWCJfQAP", "a037F00000RqujbQAB")

		cur.execute(sql, key)
		conn.commit()
		cur.close()
	
		resultMsg = "Insert statement executed!"
		
	except (Exception, psycopg2.DatabaseError) as error:
		print("Exception occured!!")
		print(error)
		resultMsg = "Database error occured."
		
	finally:
        	if conn is not None:
            		conn.close()
	return resultMsg

@route("/dbupdate")
def hello_world():

	DATABASE_URL = os.environ['DATABASE_URL']
	
	try:
		conn = psycopg2.connect(DATABASE_URL, sslmode='require')
	
		print("DB connect successfull!")

		cur = conn.cursor()
	
		sql = """ UPDATE salesforce.CampaignCandidate__c SET order__c = order__c + 1
        	        WHERE foreignkey__c = %s"""

		key = ("00001_2",)
		
		#cur.execute("SELECT firstname, lastname, email FROM salesforce.contact")
	 
		cur.execute(sql, key)
	
		conn.commit()
		cur.close()
	
		resultMsg = "Update statement executed!"
		
	except (Exception, psycopg2.DatabaseError) as error:
		print("Exception occured!!")
		print(error)
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
