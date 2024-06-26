# -*- coding: utf-8 -*-
"""
Created on 9/18/23

@author: Shahryar Doosti

Converts creator trends json files to csv file

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

def creators(path):
    with open(path, "r") as f:
        creators_json = json.load(f)
    return creators_json['trends']

def create_csv_rows(creator_dict):
    rows = []
    cid = creator_dict['creator_id']
    for item in creator_dict['points']:
        rows.append([cid, 
                     item['date'],
                     item['views'],
                     item['engagements'],
                     item['uploads'],
                     item['followers']])
    return rows

def create_csv(creator_list):
    data_list = []
    for creator in creator_list:
        data_list.extend(create_csv_rows(creator))
    return data_list
        
def create_dataframe(data_list):
    columns = ['creator_id','date','views','engagements','uploads','followers']
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

if __name__ == '__main__':
    YOGA_FILE = "final_yoga_channels_trends_091523.json" #trends_Physical Exercise_Physical Fitness_Gymnasium_Wellness_Strength Training_Pilates.json
    main(YOGA_FILE)
