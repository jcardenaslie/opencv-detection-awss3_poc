import requests
import json

conf = json.load(open("config.json"))

auth_token = conf['SOSAFE_AUTH_TOKEN']

def sosafeReportsPost(message = 'Empty'):
    
    if message is not 'Empty':
        message = "TEST @jcardenaslie {}".format(message)
    else :
        message = "TEST @jcardenaslie: {}".format(message)

    url = "https://stage-api-v3.sosafeapp.com/reports"
    hed = {'Authorization': 'Bearer ' + auth_token}
    data = {
        "type": 2,
        "latitude": "-33.4055664",
        "longitude": "-70.5681394",
        "is_anonymous": 0,
        "description": message,
        "user_id": 614228,
        "is_private": 0,
        "confirmed": 0,
        "notify_neighbours": 1,
        "latGps": "-33.4055664",
        "longGps": "-70.5681394",
        "accuracy": "16.62"
    }

    response = requests.post(url, json=data, headers=hed)
    r_json = response.json()
    # print(response.status_code)
    # print(response.json())

    if r_json['success']:
        print('{} POST success'.format(message))
    else:
        print("Error: {}".format(r_json))
