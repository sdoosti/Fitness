# -*- coding: utf-8 -*-
"""
Created on Monday 7/31/23

@author: Shahryar Doosti

It collects daily views and followers for a given list of channels

Updated to get data for general list of creators (json file)
"""


import tubular_api2 as api
import pandas as pd
import os
from time import sleep
import json

PATH = os.path.abspath(os.getcwd())
YOGA_FILE = "creators_Yoga.json"
FITENSS_FILE = "creators_Physical Exercise_Physical Fitness_Gymnasium_Wellness_Strength Training_Pilates.json"
DIET_FILE = "creators_Diet_Weight Loss.json"

def data_file_path(file_path):
    return os.path.join(PATH.replace("API","Data"),file_path)

def creator_ids(path):
    with open(path, "r") as f:
        creators_json = json.load(f)
    return creators_json['creator_ids']

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
    trend_path = path.replace("creators","trends")
    with open(trend_path, 'w') as f:
        json.dump(response, f)

def query(path):
    file_path = data_file_path(path)
    creator_list = creator_ids(file_path)
    query_post_data = post_data(creator_list)
    response = api.tubular_api('/v3.1/creator.trends',query_post_data)
    save_data(response, file_path)

query(YOGA_FILE)
sleep(60)
query(FITENSS_FILE)
sleep(60)
query(DIET_FILE)