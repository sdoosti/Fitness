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

PATH = os.path.abspath(os.getcwd())
YOGA_FILE = "final_yoga_videos_101223.json"

def video_file(file_path):
    video_path = os.path.join(PATH.replace("API","Data"),file_path)
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
        v1 = v['tvr']['v1']
        v2 = v['tvr']['v2']
        v3 = v['tvr']['v3']
        v7 = v['tvr']['v7']
        v30 = v['tvr']['v30']
        er1 = v['tvr']['e1']
        er2 = v['tvr']['e2']
        er3 = v['tvr']['e3']
        er7 = v['tvr']['e7']
        er30 = v['tvr']['e30']
        views = v['views'] 
        engagement = v['engagements']['total']
        likes = v['engagements']['breakdown'][0]['likes']
        shares = v['engagements']['breakdown'][0]['shares']
        comments = v['engagements']['breakdown'][0]['comments']
        keywords = v['keywords']        
        if v['topics'] is None:
            topics = []
        else:
            topics = [x['topic_name'] for x in v['topics']]
        rows.append([video_id,title,creator,creator_id,publish_date,duration,v1,v2,v3,v7,v30,
                     er1,er2,er3,er7,er30,views,engagement,likes,shares,comments,topics,keywords])
    return pd.DataFrame(rows, columns = ["video_id","title","creator","creator_id","publish_date",
                                         "duration","v1","v2","v3","v7","v30","e1","e2","e3","e7","e30",
                                         "views","enagement","likes","shares","comments","keywords","topics"])


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
    videos_path = os.path.join(PATH.replace("API","Data"),f"videos_{today_str}.csv")
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