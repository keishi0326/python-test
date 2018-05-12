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
	
	return '''empty param : {empty},  storeID param : {storeID}'''.format(empty=empty, storeID=storeID) 

@route("/observation-result2", method='GET')
def hello_world():
	empty = request.query.get('empty')
	storeID = request.query.get('storeID')

	#if input parameter is nothing, 'empty' literal set
	empty = "null!!" if empty is None else empty
	storeID = "null!!" if storeID is None else storeID	
	
	debug_msg = '''empty param : {empty},  storeID param : {storeID}'''.format(empty=empty, storeID=storeID) 
	print(debug_msg)

	conn, cur = db_connect()
	
	##emptyが閾値を超えているかどうかをチェック
	#パラメタで渡された店舗コードをキーに店舗マスタを検索

	#観測情報(H)へのレコード登録処理
	try:	
		cur.execute("SELECT ObservationID__c FROM salesforce.ObservationH__c order by ObservationID__c desc")
	 
	#最大のIDを取得
		for row in cur:
			maxID = row[0]
			print("maxID:{0}".format(maxID))

#		sql = """ INSERT INTO salesforce.ObservationH__c(ObservationID__c, ObservationTime__c, Availability__c, StoreSFID__c) VALUES (%s, %s, %s, %s)"""
#		key = ("OH0000001", "2018-05-12 21:26:34", "0.75", "a037F00000RqujbQAB")
#		cur.execute(sql, key)
		conn.commit()

#マスタ登録値を取得してパラメタ(empty)がその値を超えていれば

#処理を継続。超えていなければ処理終了
#観測情報と店舗マスタ、天候情報、施策マスタを結合した状態でレコード取得
#	select * from salesforce.ObservationH__c ob inner join salesforce.Store__c  store on 
#	ob.StoreSFID__c = store.id


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
	
	conn, cur = db_connect()	
	try:
		sql = """ INSERT INTO salesforce.CampaignCandidate__c(CampaignCandidateID__c, cm1_sfid__c, cm2_sfid__c, cm3_sfid__c, cm4_sfid__c, cm5_sfid__c, storesfid__c) VALUES (%s, %s, %s, %s, %s, %s, %s)"""

		key = ("CK-00000014", "a027F00000JWCIwQAP", "a027F00000JWCJQQA5", "a027F00000JWCJVQA5", "a027F00000JWCJaQAP", "a027F00000JWCJfQAP", "a037F00000RqujbQAB")

		cur.execute(sql, key)
		conn.commit()
		cur.close()
	
		resultMsg = "Insert statement executed!"
		
	except (Exception, psycopg2.DatabaseError) as error:
		print("Exception occured!!")
		print(error)
		resultMsg = "Insert error occured."
		
	finally:
        	if conn is not None:
            		conn.close()
	return resultMsg

@route("/observationinsert", method="GET")
def hello_world():

	id = request.query.get('key')

	id = id if id is not None else "00000001"
	
	conn, cur = db_connect()	
	
	try:
		sql = """ INSERT INTO salesforce.ObservationH__c(ObservationID__c, ObservationTime__c, Availability__c, StoreSFID__c) VALUES (%s, %s, %s, %s)"""

		key = ("OH0000001", "2018-05-12 21:26:34", "0.75", "a037F00000RqujbQAB")

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
	conn, cur = db_connect()	
	try:
		sql = """ UPDATE salesforce.CampaignCandidate__c SET order__c = order__c + 1 WHERE foreignkey__c = %s"""
		key = ("00001_2",)
		 
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

# DB接続、カーソル取得処理
def db_connect():
		DATABASE_URL = os.environ['DATABASE_URL']
		try:
			conn = psycopg2.connect(DATABASE_URL, sslmode='require')
			cur = conn.cursor()
		except (Exception, psycopg2.DatabaseError) as error:
			print("Exception occured!!")
			print(error)
			raise Exception("Database error occured.")

		print("DB connect successfull!")
		return conn, cur
	
run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
