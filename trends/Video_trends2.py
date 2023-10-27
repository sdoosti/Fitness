# -*- coding: utf-8 -*-
"""
Created on Monday 9/15/23

@author: Shahryar Doosti

It collects daily views for a given list of videos

Updated to get data for general list of creators (json file)
version 2: creates a batch of 200 videos
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
        values for min:  "last_day", "last_week", "last_30", "last_90", "last_1_y", "last_3_y", "all_time"
    """
    if convert_to_youtube_id:
        videos = [f"ytv_{x}" for x in videos]

    data = {
    "query": {
        "include_filter": {
            "video_gids": videos,
            #"video_upload_date": video_upload_date
        }
    },
    "date_range": date_range,
    # get 200 results per page (max available)
    'scroll': {
        'scroll_size': 200
    }
    }
    return data

    
def save_data(response, path):
    trend_path = os.path.join(PATH.replace("API","Data"),path)
    with open(trend_path, 'w') as f:
        json.dump(response, f)

def check_ids(input_ids, source_ids):
    counter = 0 
    for input_id in input_ids:
        if input_id not in source_ids:
            counter+=1
    return counter

def query(path):
    file_path = data_file_path(path)
    video_list = video_ids(file_path)
    
    results = []
    query_counter = 0
    batch_size = 50
    # actually pulled videos
    pulled = 0
    query_counter = 0 
    vids = []
    for i in range(0,len(video_list),batch_size):
        query_counter += 1
        videos = video_list[i:i+batch_size]
        query_post_data = post_data(videos,convert_to_youtube_id=True,date_range={'min':"earliest"})
        response = api.tubular_api('/v3.1/video.trends',query_post_data)
        results.extend(response.get('trends'))
        _vids = [x['id'] for x in response.get('trends')]
        vids.append(_vids)
        videos_lost = check_ids(videos, _vids )
        if videos_lost > 0:
            print(f'{videos_lost} videos lost')
        print(f'Pulled {i+batch_size} videos')
        if query_counter % 10 == 0:
            sleep(60)
    save_data(results, f"final_yoga_videos_tends_{today_str}.json")
    return results

if __name__ == "__main__":
    response = query(YOGA_FILE)
    print(response.keys())

