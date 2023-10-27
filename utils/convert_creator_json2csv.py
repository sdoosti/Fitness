# -*- coding: utf-8 -*-
"""
Created on 8/8/23

@author: Shahryar Doosti

Converts creator json files (pulled from api) to csv file
"""


import pandas as pd
import os
import json

PATH = os.path.abspath(os.getcwd())
YOGA_FILE = "creators_Yoga.json"
FITENSS_FILE = "creators_Physical Exercise_Physical Fitness_Gymnasium_Wellness_Strength Training_Pilates.json"
DIET_FILE = "creators_Diet_Weight Loss.json"


def data_file_path(file_path):
    return os.path.join(PATH.replace("API","Data"),file_path)

def creators(path):
    with open(path, "r") as f:
        creators_json = json.load(f)
    return creators_json['creators']

def create_csv_row(creator_dict):
    for account in creator_dict['accounts']:
        if account['platform'] == 'youtube':
            gid = account['gid']
            break
    return([creator_dict['creator_id'],
            creator_dict['title'],
            creator_dict['type'],
            gid,
            creator_dict['country'],
            creator_dict['language'],
            creator_dict['genre']['title'],
            creator_dict['industry']['title'],
            creator_dict['uploads_90d'],
            creator_dict['views'],
            creator_dict['last_upload_date']])

def create_csv(creator_list):
    data_list = []
    for creator in creator_list:
        data_list.append(create_csv_row(creator))
    return data_list
        
def create_dataframe(data_list):
    columns = ['creator_id','title','type','youtube_id','country','language',
               'genre','industry','uploads_90d','views','last_upload_date']
    return pd.DataFrame(data = data_list, columns = columns)

def save_data(df, path):
    csv_path = path.replace(".json",".csv")
    print(csv_path)
    DATA_PATH = os.path.join(PATH.replace("API","Data"),csv_path)
    df.to_csv(csv_path, index=False)

def main(file):
    data_path = data_file_path(file)
    creator_dict = creators(data_path)
    creator_list = create_csv(creator_dict)
    creator_df = create_dataframe(creator_list)
    save_data(creator_df,data_path)

main(YOGA_FILE)
main(FITENSS_FILE)
main(DIET_FILE)
