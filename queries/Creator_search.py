# -*- coding: utf-8 -*-
"""
Created on Monday 7/31/23

@author: Shahryar Doosti

Finding relevant creators in Fitness category
"""


import tubular_api2 as api
import pandas as pd
import os
from time import sleep
import json

PATH = os.path.abspath(os.getcwd())
# FILE_NAME = 
# DATA_PATH = os.path.join(PATH.replace("API","Data"),FILE_NAME)

def make_post_data(theme):
    """creates the query post data for search in health and fitness
    creator_genre 19: Health, Fitness & Self-Help
    creator_industry 14: Personal Car and Health"""
    if type(theme) == list:
        theme_search = " OR ".join(theme)
    else:
        theme_search = theme
    query = {
    "query": {
        "include_filter": {
            "search": theme_search,
            "creator_platforms": ["youtube"],
            "creator_genres": [19],
            "creator_languages": ["en"],
            "creator_industries": [14]
            }
            },
    "fields": [
        "title",
        "creator_id",
        "description",
        "type",
        "country",
        "language",
        "genre",
        "industry",
        "properties",
        "direct_property",
        "uploads_90d",
        "views",
        "last_upload_date",
        "accounts.title",
        "accounts.account_url",
        "accounts.platform",
        "accounts.category",
        "accounts.description",
        "accounts.gid",
        "accounts.influencer_score",
        "accounts.views",
        "accounts.monthly_views_growth",
        "accounts.monthly_v30",
        "accounts.monthly_er30",
        "accounts.audience_ages",
        "accounts.audience_locations",
        "accounts.audience_genders"
        ],
    # "sort": {
    #     "sort": "views",
    #     "sort_reverse": True
    #     },
    "scroll": {
        "scroll_size": 200
        }
    }
    return query

def save_data(response,terms):
    file_name = "creators_" + "_".join(terms) + ".json"
    print(file_name)
    DATA_PATH = os.path.join(PATH.replace("API","Data"),file_name)
    with open(DATA_PATH, "w") as f:
        json.dump(response, f)
    print(f"{file_name} is saved.")

def get_creators_by_term(theme):
    post_data = make_post_data(theme)
    creator_response = api.tubular_api('/v3/creator.search', post_data)
    save_data(creator_response, theme)

Terms_yoga = [
    "Yoga"
]

Terms_fitness = [
    "Physical Exercise",
    "Physical Fitness",
    "Gymnasium",
    "Wellness",
    "Strength Training",
    "Pilates"
]
Terms_diet = [
    "Diet",
    "Weight Loss",
]

get_creators_by_term(Terms_yoga)
sleep(60)
get_creators_by_term(Terms_fitness)
sleep(60)
get_creators_by_term(Terms_diet)