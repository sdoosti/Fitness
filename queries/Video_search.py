# -*- coding: utf-8 -*-
"""
Created on Monday 9/15/23

@author: Shahryar Doosti

Finding videos published for the given channels
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

PATH = os.path.abspath(__file__)
YOGA_FILE = "final_channel_list.csv"

def data_file(file_path):
    """ Returns the creator ids (Tubular id) for a csv file """
    full_path = os.path.join(PATH.replace("API","Data"),file_path)
    return pd.read_csv(full_path)['tubular_id'].to_list()

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
                "min": "last_3_y"
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
        "engagements_gain",
        "views",
        "engagements",
        "tvr",
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
    videos_path = os.path.join(PATH.replace("API","Data"),path)
    print(videos_path)
    with open(videos_path, 'w') as f:
        json.dump(response, f)


def get_videos(path, max_results = 1000):
    """
    pulling videos with scroll option for the last three years
    path: 
    max_results: pull at most [1000] videos
    """
    creator_list = data_file(path)
    query = make_post_data(creator_list)
    response = api('/v3/video.search',query)
    # store all videos here
    results = []
    query_counter = 0

    # actually pulled videos
    pulled = 0

    # iterate over pages until pulling all available results or max of 1000 videos
    while True:
        query_counter += 1
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

        if query_counter % 59 == 0:
            sleep(60)

    save_data(results, f"final_yoga_videos_{today_str}.json")
    return results

results = get_videos(YOGA_FILE, max_results=120000)

print(len(results))