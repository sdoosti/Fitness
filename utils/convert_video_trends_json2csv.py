# -*- coding: utf-8 -*-
"""
Created on 10/11/23

@author: Shahryar Doosti

Converts video trends json files to csv file

"""


import pandas as pd
import os
import json
from datetime import date

today = date.today()
today_str = today.strftime("%m%d%y")

#TODO #integrate the saving procedure in the api module

PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def data_file_path(file_path):
    return os.path.join(os.path.dirname(PATH),"Data",file_path)

def videos(path):
    with open(path, "r") as f:
        videos_json = json.load(f)
    return videos_json

def create_csv_rows(video_dict):
    rows = []
    vid = video_dict['id']
    for item in video_dict['points']:
        rows.append([vid, 
                     item['date'],
                     item['views'],
                     item['engagements']])
    return rows

def create_csv(video_list):
    data_list = []
    for video in video_list:
        data_list.extend(create_csv_rows(video))
    return data_list
        
def create_dataframe(data_list):
    columns = ['video_id','date','views','engagements']
    return pd.DataFrame(data = data_list, columns = columns)

def save_data(df, path):
    csv_path = path.replace(".json",".csv")
    print(csv_path)
    DATA_PATH = os.path.join(os.path.dirname(PATH),"Data",csv_path)
    df.to_csv(csv_path, index=False)

def main(file):
    data_path = data_file_path(file)
    video_dict = videos(data_path)
    video_list = create_csv(video_dict)
    video_df = create_dataframe(video_list)
    save_data(video_df,data_path)

if __name__ == '__main__':
    YOGA_FILE = "final_yoga_videos_tends_101023.json"
    main(YOGA_FILE)
