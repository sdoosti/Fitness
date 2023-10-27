# -*- coding: utf-8 -*-
"""
Created on Monday 9/15/23

@author: Shahryar Doosti

It collects daily views for a given list of videos

Updated to get data for general list of creators (json file)
"""


import tubular_api2 as api
import pandas as pd
import os
from time import sleep
import json
from datetime import date

today = date.today()
today_str = today.strftime("%m%d%y")

PATH = os.path.abspath(os.getcwd())
YOGA_FILE = "final_yoga_videos_091523.json"

def data_file_path(file_path):
    return os.path.join(PATH.replace("API","Data"),file_path)

def video_ids(path):
    with open(path, "r") as f:
        videos_json = json.load(f)
    return [x['video_id']['id'] for x in videos_json]

def post_data(videos, convert_to_youtube_id=False, 
              video_upload_date={"min": "2020-09-1","max": "2023-09-16"},
              date_range={"min": "earliest", "max": "earliest+30d"}):
    """
    returns daily trends for videos (views, engagement, ...)
    videos (list): list of videos
        youtube id should be formatted as ytv_videoid
    convert_to_youtube_id (bool): to convert video ids to youtube ids
    video_upload_date (dict): includes keys min, max indicating the date range for videos
    date_range (dict): includes keys min, max for the response date range.
        values for min:  "last_day", "last_week", "last_30", "last_90", "last_1_y", "last_3_y"
    """
    if convert_to_youtube_id:
        videos = [f"ytb_{x}" for x in videos]

    data = {
    "query": {
        "include_filter": {
            "video_gids": videos,
            "video_upload_date": video_upload_date
        }
    },
    "date_range": date_range
    }
    return data

    
def save_data(response, path):
    trend_path = os.path.join(PATH.replace("API","Data"),path)
    with open(trend_path, 'w') as f:
        json.dump(response, f)

def query(path):
    file_path = data_file_path(path)
    video_list = video_ids(file_path)
    query_post_data = post_data(video_list,date_range={'min':'last_3_y'})
    response = api.tubular_api('/v3.1/video.trends',query_post_data)
    save_data(response, f"final_yoga_videos_tends_{today_str}.json")
    return response

if __name__ == "__main__":
    response = query(YOGA_FILE)
    print(response.keys())

