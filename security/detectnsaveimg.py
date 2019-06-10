import numpy as np
import cv2
import os,shutil
import dlib

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0) # Create Camera Object(cam)
datacount=0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def facedetection():
    global datacount
    print("얼굴 인식을 위한 얼굴 데이터를 구성합니다. . ")
    train_name=input("이름을 입력하세요:")
    print("%s의 얼굴 인식 데이터생성을 시작합니다"%train_name)
    if os.path.exists("train-images/%s/"%train_name):
        shutil.rmtree("train-images/%s/"%train_name)
    os.makedirs("train-images/%s/"%train_name)
    while True:
        _,frame = cam.read()
        #getfaceThread = threading.Thread(target=getface, args=(frame,))
        #getfaceThread.start() 
        #getfaceThread.join() #wait ROI frame data for getface()
        flag,ret=getface(frame)
        if flag == 1:                       
            cv2.imwrite( "train-images/%s/frontface%d.jpg"%(train_name,datacount) ,ret)
            datacount +=1
        elif datacount>50 : 
            print("데이터 수집이 종료되었습니다.")
            datacount=0
            return
        else: #face detect failed.
            print("얼굴이 인식되지 않습니다!",end='\r',flush=True)
        cv2.imshow("cam",ret)
        cv2.waitKey(10)


def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates
        
def getface(frame):    
    # camera type is VideoCapture
    # just show cam image and return current 1 frame        
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # image binary
    faces = face_cascade.detectMultiScale(grayframe, 1.3, 5)
    detected_faces = detector(grayframe, 1) # dlib 기반 detector
    for rect in detected_faces: # i =person ,rect=which
        #frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        x = rect.left()
        y = rect.top()
        w = rect.right() - rect.left()
        h = rect.bottom() - rect.top()

        cropframe = frame[y:y+h,x:x+w]
        cropframe= cv2.resize(cropframe,(96,96), interpolation=cv2.INTER_AREA)
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
