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
import os
import json
from datetime import date

today = date.today()
today_str = today.strftime("%m%d%y")

#TODO #integrate the saving procedure in the api module

PATH = os.path.abspath(os.getcwd())

def data_file_path(file_path):
    return os.path.join(PATH.replace("API","Data"),file_path)

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
    # return([creator_dict['creator_id'],
    #         creator_dict['title'],
    #         creator_dict['type'],
    #         gid,
    #         creator_dict['country'],
    #         creator_dict['language'],
    #         creator_dict['genre']['title'],
    #         creator_dict['industry']['title'],
    #         creator_dict['uploads_90d'],
    #         creator_dict['views'],
    #         creator_dict['last_upload_date']])

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

if __name__ == '__main__':
    YOGA_FILE = "final_yoga_channels_info_091823.json"
    main(YOGA_FILE)
