# -*- coding: utf-8 -*-
"""
Created on Wed May 16 23:38:09 2018

@author: kato
"""
import sys
import urllib
import json

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
    sys.exit()
    
    #jsonの階層の"Result"以下を辞書にする。keyは番号：その次の配列がvalueになっている
    item_list = data["ResultSet"]["0"]["Result"]

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

if __name__ == '__main__':
    json_str = web_service_call()
    do_json(json_str)