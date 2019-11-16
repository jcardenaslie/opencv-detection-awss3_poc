import boto3
from botocore.client import Config
# import aiobotocore
import asyncio

import threading
import time
import os
import sosafe_api as sosafe

import json

conf = json.load(open("config.json"))

ACCESS_KEY_ID = conf['ACCESS_KEY_ID']
ACCESS_SECRET_KEY = conf['ACCESS_SECRET_KEY']
BUCKET_NAME = conf['BUCKET_NAME']

def upload(f_name):
	print('[INFO] Starting video upload thread')
	t = threading.Thread(
		target=threadUpload,
		args=(f_name, )
		)
	
	t.start()

def threadUpload(f_name):
	print('[INFO] S3 Video upload START')

	path = f_name
	data = open(path, 'rb')
	
	s3 = boto3.resource(
		's3',
		aws_access_key_id = ACCESS_KEY_ID,
		aws_secret_access_key = ACCESS_SECRET_KEY,
		config = Config(signature_version='s3v4')
	)

	s3.Bucket(BUCKET_NAME).put_object(Key=path, Body = data, ACL= 'public-read')
	
	link = 'https://s3-sa-east-1.amazonaws.com/sosafe.test/{}'.format(path)
	
	print('[INFO] S3 Video upload DONE : {}'.format(link))

	data.close()

	sosafe.sosafeReportsPost(link)

	os.remove(path)