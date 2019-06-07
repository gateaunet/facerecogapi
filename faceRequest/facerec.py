import threading
import mjpgweb 
import webrequest

def onWebcam():
    mjpgweb.onWebcam() #start webcam stream server
    print("start webcam")




if __name__=='__main__':
    camThread=threading.Thread(target=onWebcam)
    camThread.start()
    os.system('python3 webrequest.py')
    



