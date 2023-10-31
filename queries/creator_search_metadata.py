# -*- coding: utf-8 -*-
"""
Created on 9/18/23

@author: Shahryar Doosti

searching for metadata for channels by id

(very similar to creator_id_search)
#TODO: merge these two files
"""

import os, sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from API.tubular_api import api
import pandas as pd
from time import sleep
import json
from datetime import date

today = date.today()
today_str = today.strftime("%m%d%y")
PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def data_file(file_path):
    full_path = os.path.join(os.path.dirname(PATH),"Data",file_path)
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
                "account_snippet": True,
                "account_performance": True,
                "account_demographic_ages": True,
                "account_demographic_genders": True,
                "account_demographic_locations": True
            },
            "scroll": {
                "size": len(creators)
            }
    }
    return query

def save_data(response, path):
    path = os.path.join(os.path.dirname(PATH),"Data",path)
    with open(path, 'w') as f:
        json.dump(response, f)
    print(f"{path} is saved.")


def get_metadata(channel_files):
    creator_df = data_file(channel_files)
    creator_list = creator_df.iloc[:,1].to_list()
    query_post_data = make_post_data(creator_list)
    response = api('/v3.1/creator.search',query_post_data)
    save_data(response['creators'], f"final_yoga_channels_info_{today_str}.json")

if __name__ == '__main__':
    channels_file = "final_channel_list.csv"
    get_metadata(channels_file)
    
