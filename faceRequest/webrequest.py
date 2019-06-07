import requests
import threading
from multiprocessing.pool import ThreadPool
import time
import sys
URL = "http://192.168.219.158:80/facerec"
webcamStreamURL="192.168.219.142:8090/?action=snapshot"

def faceAuthRequest(requestParams,retJson):
	res = requests.get(URL,params=requestParams,timeout=20)
	res.status_code
	retJson= res.json()
	return


jsonParams = {'username':'parkjaehyun','stream_url':webcamStreamURL}

if __name__ == '__main__':
	retJson=0
	getRequestThread = threading.Thread(target=faceAuthRequest,args=(jsonParams,retJson,))
	getRequestThread.start()
	getRequestThread.join(timeout=21.0)
	if getRequestThread.isAlive(): # after 30sec ,thread:
		print("face Recognize Failed, please reload Face Recgnize Module..")
		sys.exit(1)
	print(retJson)
	sys.exit(0)
	
	
	
	
	





