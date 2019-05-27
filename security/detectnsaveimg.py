import numpy as np
import cv2
import threading
import multiprocessing
import os,shutil
face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascada = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')

cam = cv2.VideoCapture(0) # Create Camera Object(cam)

datacount=0


def facedetection():

    global datacount
    print("얼굴 학습을 위한 얼굴 데이터를 구성합니다")
    train_name=input("학습될 사람의 이름을 입력하세요:")
    print("%s의 얼굴 학습을 시작합니다"%train_name)
    
    if os.path.exists("train-images/%s/"%train_name):
        shutil.rmtree("train-images/%s/"%train_name)
    os.makedirs("train-images/%s/"%train_name)

    print("make dir")

    while True:
        _,frame = cam.read()
        #getfaceThread = threading.Thread(target=getface, args=(frame,))
        #getfaceThread.start() 
        #getfaceThread.join() #wait ROI frame data for getface()
        flag,ret=getface(frame)
        if flag == 1:                       
            cv2.imwrite( "train-images/%s/frontface%d.jpg"%(train_name,datacount)  ,ret)
            datacount +=1
        elif datacount>50 : 
            print("학습이 종료되었습니다.")
            datacount=0
            return
        else: #face detect failed.
            print("얼굴이 인식되지 않습니다!",end='\r',flush=True)
        cv2.imshow("cam",ret)
        cv2.waitKey(10)
    
        
def getface(frame):    
    # camera type is VideoCapture
    # just show cam image and return current 1 frame        
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # image binary
    faces = face_cascade.detectMultiScale(grayframe, 1.3, 5)
    
    for(x,y,w,h) in faces:
        # get face ROI Rect position(x,y,w,h)
        #frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        cropframe = frame[y:y+h,x:x+w]
        #roi_gray = grayframe[y:y+h, x:x+w] #draw ROI to gray color frame 
        #roi_color = frame [y:y+h, x:x+w] # draw ROI to BGR color frame    
        print("얼굴 인식 중. .[%d%%]"% ((datacount/50)*100) ,end="\r",flush=True)
        return (1,cropframe)

    return (0,frame)
     
     
def caminit():
    if cam.isOpened()==False: # cam check
        print("카메라가 인식되지 않습니다.")

def main():
    facedetection()
    
    

if __name__=='__main__':
    caminit()
    main()
