# -*- coding: utf-8 -*-
"""
Created on Monday 7/31/23

@author: Shahryar Doosti

Finding videos published for the given channels
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

def make_post_data(creators):
    """creates the query post data for search for the list of creators
    creators: creator ids (list)
    returns: query post data (dict)"""
    query = {
    "query": {
        "include_filter": {
            "video_platforms": [
                "youtube"
            ],
            "video_upload_date": {
                "min": "all_time"
            },
            "video_languages": [
                "en"
            ],
            "creators": creators #["NTXeR4VKrg"]
        }
    },
    "fields": [
        "title",
        "video_id",
        "views_gain",
        "keywords",
        "duration",
        "description",
        "publish_date",
        "topics",
        "categories",
        "entities",
        "publisher"
    ],
    # sort results by views in descending order
    'sort': {
        'sort': 'upload_date',#'views',
        'sort_reverse': True
    },
    # get 200 results per page (max available)
    'scroll': {
        'scroll_size': 200
    }
    }
    return query

def save_data(response, path):
    videos_path = path.replace("creators","videos")
    print(videos_path)
    with open(videos_path, 'w') as f:
        json.dump(response, f)


def get_videos(path, max_results = 1000):
    """
    pulling videos with scroll option for the last three years
    path: 
    max_results: pull at most [1000] videos
    """
    file_path = data_file_path(path)
    creator_list = creator_ids(file_path)
    query = make_post_data(creator_list)
    response = api.tubular_api('/v3/video.search',query)
    # store all videos here
    results = []

    # actually pulled videos
    pulled = 0

    # iterate over pages until pulling all available results or max of 1000 videos
    while True:
        videos = response.get('videos')

        if not videos:  # no more videos found
            print('No more videos found')
            break

        results.extend(videos)
        print('Pulled {} videos'.format(len(results)))

        if len(results) >= max_results:
            print(f'Finished pulling {max_results} videos')
            break

        # update scroll token to get the next page of results
        query['scroll']['scroll_token'] = response.get('scroll_token')
        response = api.tubular_api('/v3/video.search', query)

    save_data(results, file_path)
    return results

results = get_videos(YOGA_FILE)



print(len(results))