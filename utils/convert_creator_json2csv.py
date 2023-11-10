# -*- coding: utf-8 -*-
"""
Created on 9/18/23

@author: Shahryar Doosti

Converts creator json files (pulled from api) to csv file

changes:
    - takes the file name as an input 
    - added the creation date to the file name (not added)
    - changed the function to match the json files (for some reasons the format is different with previous files)
        figured out the problem: the function is designed for creator info not trends.
        but I had to change some of the functions as I used the more recent endpoint v3.1 instead of v3

"""


import pandas as pd
import json
from datetime import date
import os

today = date.today()
today_str = today.strftime("%m%d%y")

#TODO #integrate the saving procedure in the api module

PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def data_file_path(file_path):
    return os.path.join(os.path.dirname(PATH),"Data",file_path)

def creators(path):
    with open(path, "r") as f:
        creators_json = json.load(f)
    return creators_json

def create_csv_row(creator_dict):
    """
    for k, v in creator_dict['accounts']['youtube'].items():
    print(k)
    if type(v) == dict:
        for k2, v2 in v.items():
            print("    ",k2)
            if type(v2) == dict:
                for k3, v3 in v2.items():
                    print("        ",k)
                    
    """
    gid = creator_dict['accounts']['youtube']['gid']
    # for account in creator_dict['accounts']:
    #     if account['platform'] == 'youtube':
    #         gid = account['gid']
    #         break
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

def create_demographics_csv(creator_dict):
    rows = []
    for creator in creator_dict:
        gid = creator['accounts']['youtube']['gid']
        if 'demographics' not in creator['accounts']['youtube']:
            rows.append([gid, None, None, None, None, None, None, None, None, None, None])
            continue

        if creator['accounts']['youtube']['demographics']['ages'] is not None:
            ages_13_17 = creator['accounts']['youtube']['demographics']['ages'].get('ages_13_17', None)
            ages_18_24 = creator['accounts']['youtube']['demographics']['ages'].get('ages_18_24', None)
            ages_25_34 = creator['accounts']['youtube']['demographics']['ages'].get('ages_25_34', None)
            ages_35_44 = creator['accounts']['youtube']['demographics']['ages'].get('ages_35_44', None)
            ages_45_54 = creator['accounts']['youtube']['demographics']['ages'].get('ages_45_54', None)
            ages_55_plus = creator['accounts']['youtube']['demographics']['ages'].get('ages_55_plus', None)
        else:
            ages_13_17 = None
            ages_18_24 = None
            ages_25_34 = None
            ages_35_44 = None
            ages_45_54 = None
            ages_55_plus = None

        if creator['accounts']['youtube']['demographics']['genders'] is not None:
            female = creator['accounts']['youtube']['demographics']['genders'].get('female', None)
            male = creator['accounts']['youtube']['demographics']['genders'].get('male', None)
        else:
            female = None
            male = None
        if creator['accounts']['youtube']['demographics']['locations'] is not None:
            US = creator['accounts']['youtube']['demographics']['locations'].get('US', None)
            CA = creator['accounts']['youtube']['demographics']['locations'].get('CA', None)
        else:
            US = None
            CA = None
        row = [gid, ages_13_17, ages_18_24, ages_25_34, ages_35_44, ages_45_54, ages_55_plus,
                female, male, US, CA]
        rows.append(row)
    columns = ['creator_id', 'ages_13_17', 'ages_18_24', 'ages_25_34', 'ages_35_44', 'ages_45-54', 'ages_55_plus',
               'female', 'male', 'US', 'CA']
    df = pd.DataFrame(data=rows, columns=columns)
    df.to_csv(os.path.join(os.path.dirname(PATH),"Data","demographics.csv"), index=False)

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
    DATA_PATH = os.path.join(os.path.dirname(PATH),"Data",csv_path)
    df.to_csv(csv_path, index=False)

def main(file):
    data_path = data_file_path(file)
    creator_dict = creators(data_path)
    creator_list = create_csv(creator_dict)
    creator_df = create_dataframe(creator_list)
    save_data(creator_df,data_path)
    create_demographics_csv(creator_dict)

if __name__ == '__main__':
    YOGA_FILE = "final_yoga_channels_info_091823.json"
    main(YOGA_FILE)
