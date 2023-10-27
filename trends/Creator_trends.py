# -*- coding: utf-8 -*-
"""
Created on Monday 7/31/23

@author: Shahryar Doosti

It collects daily views and followers for a given list of channels (csv file)
"""


import tubular_api2 as api
import pandas as pd
import os
from time import sleep
import json

PATH = os.path.abspath(os.getcwd())
FILE_NAME = "channel_ids.csv"
NEW_FILE = "channel_ids_updated.csv"
DATA_PATH = os.path.join(PATH.replace("API","Data"),FILE_NAME)

channels_df = pd.read_csv(DATA_PATH)
#channels = channels_df.channel_id.tolist()

def channel_search(channel):
    creator_response = api.tubular_api('/v3/creator.search', {
        'query': {        
            'include_filter': {'search': channel,
                               'creator_platforms': ['youtube']
                            #    'accounts' : [{
                            #        'platform' : "youtube"
                            #    }]
            }
                },    
        'fields': [
            'creator_id',        
            'title',
            ]
        })
    if not creator_response.get('creators'):
        print('No creators found')
    else:
        for creator in creator_response.get('creators'): 
            if creator['title'] == channel:
                return creator['creator_id'], creator['title']
        for creator in creator_response.get('creators'):
            return creator['creator_id'], creator['title']

channel_ids = []
channel_titles = []
channel_dict = {}
for k,channel in channels_df.iterrows():
    try:
        cid, title = channel_search(channel.channel_name)
    except:
        cid, title = channel_search("Roberta's Gym")
    print(k, title)
    channel_dict[title] = cid
    channel_ids.append(cid)
    channel_titles.append(title)
    sleep(1)

channels_df['yt_channel_id'] = channel_ids
channels_df['channel_title'] = channel_titles

NEW_DATA_PATH = os.path.join(PATH.replace("API","Data"),NEW_FILE)

channels_df.to_csv(NEW_DATA_PATH,index = False)

data = {
    "creator_ids" : channel_ids,
    "platforms" : ["youtube"],
    "metrics" : ["views", "engagements", "uploads", "followers"],
    "date_range" : {
        "min" : "last_3_y"},
    "time_bucket" : "days"
}


response = api.tubular_api('/v3.1/creator.trends',data)

with open(os.path.join(PATH.replace("API","Data"),"trends.json"), 'w') as f:
    json.dump(response, f)
