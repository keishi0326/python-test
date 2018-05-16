import os
from bottle import route, run, get, post, request
import sys
import time,datetime
import pandas as pd
import pandas.io.sql as psql
from sklearn import svm, grid_search, cross_validation, metrics
#from sklearn.ensemble         import RandomForestRegressor
from sklearn.ensemble         import GradientBoostingRegressor
import psycopg2 #DB Connect

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

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

	#if input parameter is nothing, fix literal set
	empty = "75" if empty is None else empty
	storeID = "a037F00000RqujbQAB" if storeID is None else storeID
	
	debug_msg = '''empty param : {empty},  storeID param : {storeID}'''.format(empty=empty, storeID=storeID)
	print(debug_msg)

	conn, cur = db_connect()

	try:
		cur.execute("SELECT ObservationID__c FROM salesforce.ObservationH__c order by ObservationID__c desc")

		for row in cur:
			maxID = row[0]
			print("Current maxID:{0}".format(maxID))
			break
			
		#extract number part
		substr = maxID[2:]
		#number increment and concatenate prefix"OH"
		newID = "OH{:07}".format(int(substr) + 1)
		
		newID = "OH0000001" if newID is None else newID
		
		now = datetime.datetime.now()

#		sql = """ INSERT INTO salesforce.ObservationH__c(ObservationID__c, ObservationTime__c, Availability__c, StoreSFID__c) VALUES (%s, %s, %s, %s)"""
#		key = ("OH0000001", "2018-05-12 21:26:34", "0.75", "a037F00000RqujbQAB")
#		cur.execute(sql, key)

		empty = int(float(empty))

		sql = """ INSERT INTO salesforce.ObservationH__c(ObservationID__c, ObservationTime__c, Availability__c, StoreSFID__c) VALUES (%s, %s, %s, %s)"""
		key = (newID, now, str(empty), storeID)
		cur.execute(sql, key)
		conn.commit()
		
#店舗マスタ登録値を取得してパラメタ(empty)がその値を超えていれば
#処理を継続。超えていなければ処理終了
		sql_store = "select congestionjudgevalue__c, zip__c from salesforce.Store__c where sfid = %s"""
		key_store = (storeID,)
		cur.execute(sql_store, key_store)

		row = cur.fetchone()
		judge_value = row[0]
		zip = row[1]
		
		if empty < judge_value:
			return "The observed situation is crowded!"
	
		print("Recommendation process starts!!")

		# 暫定処理　現在の天候データレコードを追加
		# 追加レコードが必要かどうか確認
		sql_weather = "select WeatherPrimaryKey__c, Zip__c, to_char(ObservationTime__c, 'YYYYMMDDHH24') from salesforce.WeatherInfo__c where Zip__c = %s and to_char(ObservationTime__c, 'YYYYMMDDHH24') = %s"""
		where_clause = (zip, now.strftime('%Y%m%d%H'))
		cur.execute(sql_weather, where_clause)
		row = cur.fetchone()
		
		# もし、既に天候レコードが存在していた場合には、天候レコード追加処理をスキップ
		if row is None:
			print("Current weather record does not exist! Record weather create process starts!") 
			# キー値の取得
			sql_weather = "select WeatherPrimaryKey__c from salesforce.WeatherInfo__c order by WeatherPrimaryKey__c desc"""
			cur.execute(sql_weather)
			row = cur.fetchone()
			primary_key = "00000001" if row is None else "{:08}".format(int(row[0]) +1) 
			print("weather new_key: " + primary_key) 
			
			# 天候情報の追加
			sql = """ INSERT INTO salesforce.WeatherInfo__c(Zip__c, Temparature__c, ObservationTime__c, Weather__c, WeatherPrimaryKey__c) VALUES (%s, %s, %s, %s, %s)"""
			# 最終実装は、天気APIから天気情報を取得してセット
			key = (zip, 20, now, "晴れ", primary_key)
			cur.execute(sql, key)
			conn.commit()		
		else:
			print("Current weather record exists! Create Process skips.") 

		sql_target2 = "select weather.weather__c as weather, store.locationrequiremen__c as segment, cam.campaignid__c as campaign_id, ob.observationtime__c as datetime from salesforce.ObservationH__c ob inner join salesforce.Store__c  store on " \
		"ob.StoreSFID__c = store.sfid " \
		"  left join salesforce.WeatherInfo__c weather on store.Zip__c = weather.Zip__c and " \
		"  to_char(ob.observationtime__c, 'YYYYMMDDHH24') = to_char(weather.observationtime__c, 'YYYYMMDDHH24') " \
		"  inner join  salesforce.CampaignMaster__c cam on true = true " \
		" where ob.ObservationID__c = %s"
		key_target2 = (newID, )

#		df = psql.read_sql(sql_target2, conn, params=key_target2, parse_dates = [1])
		df = psql.read_sql(sql_target2, conn, params=key_target2)
		
#		print(df.loc[0])
#		print(df)
		
		predict(df)
		
		# 暫定処理
		insert_campaign(conn, cur)
		
	except (Exception, psycopg2.DatabaseError) as error:
		print("Exception occured!!")
		print(error)
		resultMsg = "DB error occured."		
	finally:
        	if conn is not None:
            		conn.close()

	return "process succeeded!"

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

# リコメンデーション処理
def predict(df):
	df['year'] = df['datetime'].dt.strftime('%Y')
	df['month'] = df['datetime'].dt.strftime('%m')
	df['weekday'] = df['datetime'].dt.strftime('%w')
	df['day'] = df['datetime'].dt.strftime('%d')

	df['time'] = df['datetime'].dt.strftime('%H')
	
	print("datafrme first row:")
	print(df.loc[0])

	le_seg = LabelEncoder().fit(["ビジネス", "住宅", "学校", "観光","駅周辺"])
	le_weather = LabelEncoder().fit(["晴れ", "曇り", "雨", "雪", "暴風"])

	df['segment'] = le_seg.transform(df['segment'])
	df['weather'] = le_weather.transform(df['weather'])

	data = df.drop(["datetime"], axis=1)

	print(data)

	# 学習した分類器を読み込む。
	classifier = joblib.load('data/model_v2.pkl')

	# パラメータを表示してみる。
	print (classifier)

	result = classifier.predict(data)

	#df_date['result'] = result
	#df_result = pd.DataFrame()
	#df_result[1]=result

	df_result = data
	df_result["result"] = result

	print(df_result)
	
	df_result.sort_values(by=["result"], ascending=False)

	print(df_result)

	
@route("/createmodel")
def hello_world():
	df = pd.read_csv('data/promotion.csv',sep=',',encoding='SHIFT-JIS', parse_dates = [2,3])

	df = transform(df)
	
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

	# 学習した分類器を保存する。
	joblib.dump(clf_cv, 'data/model_v2.pkl', compress=True)
		
	
#  キャンペーン候補追加
def insert_campaign(conn, cur):
	cur.execute("SELECT campaigncandidateid__c FROM salesforce.CampaignCandidate__c order by campaigncandidateid__c desc")

	for row in cur:
		maxID = row[0]
		print("Current maxID:{0}".format(maxID))
		break
			
	#extract number part
	substr = maxID[3:]
	#number increment and concatenate prefix"OH"
	newID = "CK-{:08}".format(int(substr) + 1)
	
	newID = "CK-00000001" if newID is None else newID

	sql = """ INSERT INTO salesforce.CampaignCandidate__c(CampaignCandidateID__c, cm1_sfid__c, cm2_sfid__c, cm3_sfid__c, cm4_sfid__c, cm5_sfid__c, storesfid__c) VALUES (%s, %s, %s, %s, %s, %s, %s)"""

	key = (newID, "a027F00000JWCIwQAP", "a027F00000JWCJQQA5", "a027F00000JWCJVQA5", "a027F00000JWCJaQAP", "a027F00000JWCJfQAP", "a037F00000RqujbQAB")

	cur.execute(sql, key)
	conn.commit()
	cur.close()
	
	print ("Insert campaign candicate statement executed!")
	
	
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
	
def transform(df):
    df['year'] = df['date'].dt.strftime('%Y')
    df['month'] = df['date'].dt.strftime('%m')
    df['weekday'] = df['date'].dt.strftime('%w')
    df['day'] = df['date'].dt.strftime('%d')

    df['time'] = df['Time'].dt.strftime('%H')

    le_seg = LabelEncoder().fit(["ビジネス", "住宅", "学校", "観光", "駅周辺"])
    le_weather = LabelEncoder().fit(["晴れ", "曇り", "雨", "雪", "暴風"])

    df['segment'] = le_seg.transform(df['segment'])
    df['weather'] = le_weather.transform(df['weather'])    

    return df
	
run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


