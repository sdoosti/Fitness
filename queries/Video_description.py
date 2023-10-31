# -*- coding: utf-8 -*-
"""
Created on Monday 9/17/23

@author: Shahryar Doosti

It arranges video description and flags each video if a challenge is introduced
"""

import pandas as pd
from nltk.probability import FreqDist
import os
import json
from datetime import date
import re

today = date.today()
today_str = today.strftime("%m%d%y")

PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOGA_FILE = "final_yoga_videos_091523.json"

def video_file(file_path):
    video_path = os.path.join(os.path.dirname(PATH),"Data",file_path)
    with open(video_path, "r") as f:
        videos_json = json.load(f)
    return videos_json

def json2csv(video_json):
    rows = []
    for v in video_json:
        video_id = v['video_id']['id']
        title = v['title']
        creator = v['publisher']['creator_name']
        creator_id = v['publisher']['creator_id']
        publish_date = v['publish_date']
        duration = v['duration']
        keywords = v['keywords']        
        if v['topics'] is None:
            topics = []
        else:
            topics = [x['topic_name'] for x in v['topics']]
        rows.append([video_id,title,creator,creator_id,publish_date,duration,topics,keywords])
    return pd.DataFrame(rows, columns = ["video_id","title","creator","creator_id","publish_date",
                                         "duration","keywords","topics"])


def flag_challenges(text):
    """
    flags a string to determine they include keywords "challenge" or "Day #"
    text (str): video title
    return (bool) if flagged
    """
    return (len(re.search('day \d+|$', text.lower()).group()) > 0 ) | ('challenge' in text.lower())

def find_challenge_videos(video_titles):
    """
    flags videos that have challenge or day in their title
    videos (pandas.Series): a pandas series of video titles to be flagged
        each video is a dictionary
    return (pandas.Series)
    """
    return video_titles.apply(flag_challenges)

if __name__ == '__main__':
    videos_json = video_file(YOGA_FILE)
    video_df = json2csv(videos_json)
    video_df['publish_date'] = pd.to_datetime(video_df.publish_date)
    video_df.sort_values(['creator','publish_date'],inplace=True)
    video_df['challenge'] = find_challenge_videos(video_df.title)

    # how many 'challenge' videos we have per creator
    #video_df['cum'] = video_df.groupby('creator').challenge.cumsum()
    video_df.groupby('creator').challenge.sum().describe()

    creators = video_df.groupby(['creator','creator_id'])['video_id'].count().reset_index()

    # save to csv file
    videos_path = os.path.join(os.path.dirname(PATH),"Data",f"videos_{today_str}.csv")
    video_df.to_csv(videos_path, index=False)

    """ 
    fdist = FreqDist()

    for video in videos_json:
        for keyword in video['keywords']:
            for word in keyword.split():
                fdist[word.lower()] += 1

    fdist.most_common(30)

    videos_flagged = {}
    videos_flagged2 = {}
    for video in videos_json:
        flag1 = False
        flag2 = False
        if 'challenge' in video['title'].lower():
            flag1 = True
        for keyword in video['keywords']:
            if 'challenge' in keyword.lower():
                flag2 = True
                break
        if flag1 | flag2:
            videos_flagged[video['video_id']['id']] = {"desc": flag1,
                                                    "keyw": flag2}
        if flag2 and not flag1:
            videos_flagged2[video['video_id']['id']] = {"desc": video['description'],
                                                        "keywords": video['keywords'],
                                                        "title": video['title']}


    for k, v in videos_flagged2.items():
        print('-'*40)
        print(k)
        print(v['title'])
        print()
        print(v['keywords'])

    videos_flagged = {}
    cnt1 = 0
    cnt2 = 0
    for video in videos_json:
        if len(re.search('day \d+|$', video['title'].lower()).group())>0:
            cnt1 += 1
        if 'challenge' in video['title'].lower():
            cnt2 += 1 """