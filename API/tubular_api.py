# -*- coding: utf-8 -*-
"""
Created on Monday 7/31/23

@author: Shahryar Doosti
"""

from requests_oauthlib import OAuth2Session
from flask import Flask, request, redirect, session, url_for
from flask.json import jsonify
import json
import os

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

client_id = "26275-60bd83eacc1edfe7b4949999ccd167a4"
client_secret = "51ea3703e394fa60e917b37db64be4c2"
#token = {"token_type": "bearer", "access_token": "2|1:0|10:1518141264|5:token|88:eyJpZCI6MjYyNzUsImNsaWVudF9pZCI6IjI2Mjc1LTYwYmQ4M2VhY2MxZWRmZTdiNDk0OTk5OWNjZDE2N2E0In0=|eda1a5dfdc22b3a80e0f7f0050b1b9470fd60a56d57702781135fed7e8b07b8d"}  

auth_url = "https://tubularlabs.com/api/v3/oauth.token"

header = {'Content-type': 'application/json'}


tub = OAuth2Session(client_id)
token = tub.fetch_token(auth_url,code=client_secret)

class tubular(object):
    def __init__(self,client_id,client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_url = auth_url = "https://tubularlabs.com/api/v3.1/oauth.token"
        tub = OAuth2Session(self.client_id)
        self.token = tub.fetch_token(self.auth_url,code=self.client_secret)
        self.session = OAuth2Session(self.client_id,token=self.token)
    
    def test(self):
        url = "https://tubularlabs.com/api/v3/auth.test"
        response = self.session.post(url=url)
        if response.status_code == 200:
            return(True)
        else:
            return(False)
    
    def limit(self):
        url = 'https://tubularlabs.com/api/v3/rate_limit.details'
        return(self.session.get(url=url))
          
    def creator_audience_overlap(self,c_id):
        url = "https://tubularlabs.com/api/v3/creator.audience_overlap"
        if type(c_id)==list:
            data = {"creator_ids":[str(x) for x in c_id],"type": "creator","platform":"facebook"}
        elif c_id.isdigit:
            data = {"creator_ids":[str(c_id)], "type": "creator","platform":"facebook"}
        else:
            raise Exception("Creator_id should be either a number or list")
        header = {'Content-type': 'application/json'}
        return(self.session.post(url=url,data=json.dumps(data),headers=header))

    def creator_details(self,c_id):
        url = "https://tubularlabs.com/api/v3/creator.details"
        if type(c_id)==list:
            data = {"creator_ids":[str(x) for x in c_id]}
        elif c_id.isdigit:
            data = {"creator_ids":[str(c_id)]}
        else:
            raise Exception("Creator_id should be either a number or list")
        header = {'Content-type': 'application/json'}
        return(self.session.post(url=url,data=json.dumps(data),headers=header))

    def creator_facets(self,search,facets):
        # Top 50 creators based on search
        """ 
        facets:            
            "creator_types"
            "creator_platforms"
            "creator_genres"
            "creator_countries"
            "creator_languages"
            "creator_industries"
            "creator_properties"
            "creator_rising_star"
            "creator_themes"
        
        Refer to the CreatorFilter section for details on what each of these represent.
        """
        url = "https://tubularlabs.com/api/v3/creator.facets"
        data = {"query": {"include_filter": {"search":search}},"facets":facets}
        header = {'Content-type': 'application/json'}
        return(self.session.post(url=url,data=json.dumps(data),headers=header))
             
    def creator_search(self,search,ctype=None):
        url = "https://tubularlabs.com/api/v3/creator.search"
        if ctype == None:
            data = {"query": {"include_filter": {"search":search,
                                            "creator_platforms":["facebook"]}}}
        else:
            data = {"query": {"include_filter": {"search":search,
                                            "creator_types":[ctype],
                                            "creator_platforms":["facebook"]}}}
        header = {'Content-type': 'application/json'}
        return(self.session.post(url=url,data=json.dumps(data),headers=header))    

    def creator_summary(self,search):
        url = "https://tubularlabs.com/api/v3/creator.summary"
        data = {"query": {"include_filter": {"search":search}}}
        header = {'Content-type': 'application/json'}
        return(self.session.post(url=url,data=json.dumps(data),headers=header))     

    def creator_trends(self,c_id,date_range={"min":"last_90"},metrics=['views']):
        url = "https://tubularlabs.com/api/v3/creator.trends"
        data = {"creator_id": c_id, "date_range": date_range, "metrics":metrics} 
        header = {'Content-type': 'application/json'}
        return(self.session.post(url=url,data=json.dumps(data),headers=header))  

    def video_details(self,v_id):
        url = "https://tubularlabs.com/api/v3/video.details"
        if type(v_id)==list:
            data = {"video_ids":[{"id":str(x),"platform":"facebook"} for x in v_id]}
        elif v_id.isdigit:
            data = {"video_ids":[{"id": str(v_id) ,"platform":"facebook"}]}
        else:
            raise Exception("Video_id should be either a number or list")
        header = {'Content-type': 'application/json'}
        return(self.session.post(url=url,data=json.dumps(data),headers=header))        
                

if __name__ == '__main__':
    tub = tubular(client_id, client_secret)
    tub.test()
    tub.limit()
