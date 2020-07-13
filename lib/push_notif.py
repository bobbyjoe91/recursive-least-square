import requests
import json

def pushbullet_notif(title, body):
    msg = {"type": "note", "title": title, "body": body}
    TOKEN = "" # insert your token here
    resp = requests.post('https://api.pushbullet.com/v2/pushes', 
                         data=json.dumps(msg),
                         headers={'Authorization': 'Bearer ' + TOKEN,
                                  'Content-Type': 'application/json'})
    if resp.status_code != 200:
        print('Message failed to sent. Please check your internet connection.')
    else:
        print ('Message sent')