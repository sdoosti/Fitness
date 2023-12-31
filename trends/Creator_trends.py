# -*- coding: utf-8 -*-
"""
Created on Monday 9/15/23

@author: Shahryar Doosti

It collects daily views and followers for a given list of channels

Updated to get data for general list of creators (json file)
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
YOGA_FILE = "final_channel_list.csv"

def data_file(file_path):
    """ Returns the creator ids (Tubular id) for a csv file """
    full_path = os.path.join(os.path.dirname(PATH),"Data",file_path)
    return pd.read_csv(full_path)['tubular_id'].to_list()

def post_data(creators):
    data = {
        "creator_ids" : creators,
        "platforms" : ["youtube"],
        "metrics" : ["views", "engagements", "uploads", "followers"],
        "date_range" : {
            "min" : "last_3_y"},
        "time_bucket" : "days"
    }
    return data

def save_data(response, path):
    trend_path = os.path.join(os.path.dirname(PATH),"Data",path)
    with open(trend_path, 'w') as f:
        json.dump(response, f)

def query(path):
    creator_list = data_file(path)
    query_post_data = post_data(creator_list)
    response = api('/v3.1/creator.trends',query_post_data)
    save_data(response, f"final_yoga_channels_tends_{today_str}.json")
    return response

response = query(YOGA_FILE)
print(response.keys())

