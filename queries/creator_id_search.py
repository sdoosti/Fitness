# -*- coding: utf-8 -*-
"""
Created on 9/14/23

@author: Shahryar Doosti

It checks the list of YouTube Channels and search for Tubular ids
"""


import tubular_api2 as api
import pandas as pd
import os
from time import sleep
import json

PATH = os.path.abspath(os.getcwd())
channels_file = "final_channel_list.csv"

def data_file(file_path):
    full_path = os.path.join(PATH.replace("API","Data"),file_path)
    return pd.read_csv(full_path)

def make_post_data(creators):
    """creates the query post data for search for the list of creators
    creators: creator ids (list)
    returns: query post data (dict)"""
    # query = {
    #         "include": {
    #             "search": "https://www.youtube.com/c/lexfridman"
    #         },
    #         "scroll": {
    #             "size": 1
    #         }
    # }
    query = {
            "include": {
                 "creators": creators,
                 "platform": ["youtube"]
            },
            "fields": {
                "snippet": True,
                "account_snippet": True
            },
            "scroll": {
                "size": len(creators)
            }
    }
    return query

def sanity_check_creator_id(channel_files):
    creator_df = data_file(channels_file)
    creator_list = creator_df.iloc[:,1].to_list()
    query_post_data = make_post_data(creator_list)
    response = api.tubular_api('/v3.1/creator.search',query_post_data)
    for k, row in creator_df.iterrows():
        print('-'*30)
        print(k)
        print(row.channel_name)
        for item in response['creators']:
            if item['creator_id'] == row.tubular_id:
                print("xxxxxxFOUNDxxxxxx")
                print(item['title'])
                break
