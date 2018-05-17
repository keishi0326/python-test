# -*- coding: utf-8 -*-
"""
Created on Wed May 16 23:38:09 2018

@author: kato
"""
import sys
import urllib
import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_weather(zip):
    json_str = web_service_call(zip)
    do_json(json_str)
	
def web_service_call(zip):
	#https://openweathermap.org/weather-conditions
    url = 'http://api.openweathermap.org/data/2.5/weather?'
    appid = '34ab458c5035d9fe6201672346b56002'
    params = urllib.parse.urlencode(
            {'appid': appid,
             'zip':'274-0817,jp',})

    response = urllib.request.urlopen(url + params)
    return response.read()

def do_json(s):
    data = json.loads(s)
    print(json.dumps(data, sort_keys=True, indent=4))
    print( data["weather"][0]["id"])
    print( str(data["weather"][0]["id"])[0])


    hash = { "2": "暴風", "3": "霧雨", "5":"雨", "6":"雪", "7":"Atmosphere", "8":"晴れ", "9":"曇り"}

    #weather code の１桁目を取得
    weather_code = str(data["weather"][0]["id"])[0]
    print( "weather code : " + type(weather_code) + " : " + weather_code )
    
    weatehr_name = hash[weather_code]
    print( "weather name : " + weather_name )

    return weather_name
	
	
    #jsonの階層の"Result"以下を辞書にする。keyは番号：その次の配列がvalueになっている

    #空のディクショナリを作る
#    ranking = {}
#    for  k, v in item_list.iteritems():
#        try:
#            rank = int(v["_attributes"]["rank"])
#            vector = v["_attributes"]["vector"]
#            name  = v["Name"]
#            ranking[rank] = [vector, name]
#        except:
#            if k == "RankingInfo":
#                StartDate = v["StartDate"]
#                EndDate = v["EndDate"]  

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

if __name__ == '__main__':
    json_str = web_service_call()
    do_json(json_str)
