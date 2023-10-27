# -*- coding: utf-8 -*-
"""
Created on Monday 7/31/23

@author: Shahryar Doosti
"""

import requests
import json
import os

API_KEY = "58107-3eb77411277d73418bd05999791f1930"

headers = {'Content-Type': 'application/json',
          'Api-Key': API_KEY}

def test():
    url = 'https://tubularlabs.com/api/v3/auth.test'
    response = requests.post(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f'Tubular API call failed: {response.text}, \nError Code: {response.status_code}')

    print(f'API call succeeded: {response.text}')

def handle_error(response):    
    """Raise exception when HTTP code is not 200"""
    if response.status_code == 500:        
        error = 'Internal error. Please contact Tubular'
    else:        
        error = f'API call failed with code {response.status_code}: {response.text}'
    raise Exception(error)

def tubular_api(endpoint, data):    
    """General function to call Tubular API
    Args:        
        endpoint: Endpoint name including version
        data: Request body as a dictionary
    Returns:        
        response: API response as a dictionary
    """    
    http_response = requests.post(
        f'https://tubularlabs.com/api{endpoint}',        
        headers={
            'Content-Type': 'application/json',            
            'Api-Key': API_KEY
        },        
        json=data
    )

    # Unexpected error happened    
    if http_response.status_code == 500:
        return handle_error(http_response)
    # Incorrect API call, e.g. rate-limits or wrong parameters    
    if http_response.status_code != 200:
        return handle_error(http_response)
    
    response = http_response.json()    
    if not response.get('ok'):
        handle_error(http_response)

    return response

# creator_response = tubular_api('/v3/creator.search', {
#     'query': {        
#         'include_filter': {'search': 'https://www.youtube.com/@ChloeTing'}
#             },    
#     'fields': [
#         'creator_id',        
#         'title',
#         ]
#     })
# if not creator_response.get('creators'):
#     print('No creators found')
# else:
#     for creator in creator_response.get('creators'):        
#         print(f"Found a creator {creator['creator_id']} with title {creator['title']}")

