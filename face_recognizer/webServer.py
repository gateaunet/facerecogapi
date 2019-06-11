from flask import Flask
import flask
from flask_restful import Resource, Api,reqparse
from flask import request,render_template
import cv2
import glob
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.python.keras.layers.core import Lambda, Flatten, Dense
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.pooling import MaxPooling2D, AveragePooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
import time

from utils import LRN2D
import utils
import urllib
from urllib import request
import sys
import dlib
face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

predictor_model = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()

face_pose_predictor = dlib.shape_predictor(predictor_model)

def create_model(Input):
    x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(Input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Lambda(LRN2D, name='lrn_1')(x)
    x = Conv2D(64, (1, 1), name='conv2')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), name='conv3')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
    x = Activation('relu')(x)
    x = Lambda(LRN2D, name='lrn_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    # Inception3a
    inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
    inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
    inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

    inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
    inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
    inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

    inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
    inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
    inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
    inception_3a_pool = Activation('relu')(inception_3a_pool)
    inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

    inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
    inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
    inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

    inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

    # Inception3b
    inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
    inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
    inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

    inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
    inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
    inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

    inception_3b_pool = Lambda(lambda x: x ** 2, name='power2_3b')(inception_3a)
    inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
    inception_3b_pool = Lambda(lambda x: x * 9, name='mult9_3b')(inception_3b_pool)
    inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)
    inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
    inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
    inception_3b_pool = Activation('relu')(inception_3b_pool)
    inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

    inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
    inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
    inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

    inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

    # Inception3c
    inception_3c_3x3 = utils.conv2d_bn(inception_3b,
                                       layer='inception_3c_3x3',
                                       cv1_out=128,
                                       cv1_filter=(1, 1),
                                       cv2_out=256,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(2, 2),
                                       padding=(1, 1))

    inception_3c_5x5 = utils.conv2d_bn(inception_3b,
                                       layer='inception_3c_5x5',
                                       cv1_out=32,
                                       cv1_filter=(1, 1),
                                       cv2_out=64,
                                       cv2_filter=(5, 5),
                                       cv2_strides=(2, 2),
                                       padding=(2, 2))

    inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
    inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

    inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

    # inception 4a
    inception_4a_3x3 = utils.conv2d_bn(inception_3c,
                                       layer='inception_4a_3x3',
                                       cv1_out=96,
                                       cv1_filter=(1, 1),
                                       cv2_out=192,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(1, 1),
                                       padding=(1, 1))
    inception_4a_5x5 = utils.conv2d_bn(inception_3c,
                                       layer='inception_4a_5x5',
                                       cv1_out=32,
                                       cv1_filter=(1, 1),
                                       cv2_out=64,
                                       cv2_filter=(5, 5),
                                       cv2_strides=(1, 1),
                                       padding=(2, 2))

    inception_4a_pool = Lambda(lambda x: x ** 2, name='power2_4a')(inception_3c)
    inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
    inception_4a_pool = Lambda(lambda x: x * 9, name='mult9_4a')(inception_4a_pool)
    inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)
    inception_4a_pool = utils.conv2d_bn(inception_4a_pool,
                                        layer='inception_4a_pool',
                                        cv1_out=128,
                                        cv1_filter=(1, 1),
                                        padding=(2, 2))
    inception_4a_1x1 = utils.conv2d_bn(inception_3c,
                                       layer='inception_4a_1x1',
                                       cv1_out=256,
                                       cv1_filter=(1, 1))
    inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

    # inception4e
    inception_4e_3x3 = utils.conv2d_bn(inception_4a,
                                       layer='inception_4e_3x3',
                                       cv1_out=160,
                                       cv1_filter=(1, 1),
                                       cv2_out=256,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(2, 2),
                                       padding=(1, 1))
    inception_4e_5x5 = utils.conv2d_bn(inception_4a,
                                       layer='inception_4e_5x5',
                                       cv1_out=64,
                                       cv1_filter=(1, 1),
                                       cv2_out=128,
                                       cv2_filter=(5, 5),
                                       cv2_strides=(2, 2),
                                       padding=(2, 2))
    inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
    inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

    inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

    # inception5a
    inception_5a_3x3 = utils.conv2d_bn(inception_4e,
                                       layer='inception_5a_3x3',
                                       cv1_out=96,
                                       cv1_filter=(1, 1),
                                       cv2_out=384,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(1, 1),
    padding = (1, 1))

    inception_5a_pool = Lambda(lambda x: x ** 2, name='power2_5a')(inception_4e)
    inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
    inception_5a_pool = Lambda(lambda x: x * 9, name='mult9_5a')(inception_5a_pool)
    inception_5a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_5a')(inception_5a_pool)
    inception_5a_pool = utils.conv2d_bn(inception_5a_pool,
                                        layer='inception_5a_pool',
                                        cv1_out=96,
                                        cv1_filter=(1, 1),
                                        padding=(1, 1))
    inception_5a_1x1 = utils.conv2d_bn(inception_4e,
                                       layer='inception_5a_1x1',
                                       cv1_out=256,
                                       cv1_filter=(1, 1))

    inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

    # inception_5b
    inception_5b_3x3 = utils.conv2d_bn(inception_5a,
                                       layer='inception_5b_3x3',
                                       cv1_out=96,
                                       cv1_filter=(1, 1),
                                       cv2_out=384,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(1, 1),
                                       padding=(1, 1))
    inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
    inception_5b_pool = utils.conv2d_bn(inception_5b_pool,
                                        layer='inception_5b_pool',
                                        cv1_out=96,
                                        cv1_filter=(1, 1))
    inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

    inception_5b_1x1 = utils.conv2d_bn(inception_5a,
                                       layer='inception_5b_1x1',
                                       cv1_out=256,
                                       cv1_filter=(1, 1))
    inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

    av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
    reshape_layer = Flatten()(av_pool)
    dense_layer = Dense(128, name='dense_layer')(reshape_layer)
    norm_layer = Lambda(lambda x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)

    # Final Model
    model = Model(inputs=[Input], outputs=norm_layer)

    return model


# 학습용 이미지를 임베딩으로 변환하는 과정
def image_to_embedding(image, model):
    image = cv2.resize(image, (96, 96))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
    img_array = np.array([img])
    # 기학습된 모델에 이미지를 입력으로 이미지에 대한 embedding vector를 반환한다.
    embeddings = model.predict_on_batch(img_array)
    print("임베딩 변환과정중 임베딩:")
    print(embeddings)
    return embeddings

# 학습 디렉토리에있는 이미지를 128-D 임베딩 벡터로 변환하여 inpurt_embeddings 객체 반환
def create_input_image_embeddings(model):
    input_embeddings = {}
    for file in glob.glob("train-images/*"):
        #확장명 없는 순수 파일명(이름)가져오기.
        person_name = os.path.splitext(os.path.basename(file))[0]
        image = cv2.imread(file, 1)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #faces = face_cascade.detectMultiScale(gray_img, 1.2, 5)
        input_embeddings[person_name] = image_to_embedding(gray_img, model)
    return input_embeddings

def initWeights():
    # 기학습된 모델 가중치 불러오기.
    global weights
    global weights_dict
    weights = utils.weights
    weights_dict = utils.load_weights()
    # Set layer weights of the model
    for name in weights:
        if facemodel.get_layer(name) != None:
            facemodel.get_layer(name).set_weights(weights_dict[name])
        elif facemodel.get_layer(name) != None:
            facemodel.get_layer(name).set_weights(weights_dict[name])


def initModel():
    input = Input(shape=(96, 96, 3))
    model =create_model(input)
    return model



# small2.nn2 모델 생성 및 기학습된 가중치 초기화.

# 현재 얼굴 이미지를 임베딩 벡터화 시킴.
def image_to_embedding(image):
    image = cv2.resize(image, (96, 96))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
    img_array = np.array([img])
    with tf_session.as_default():
        with tf_graph.as_default():
            embedding = facemodel.predict_on_batch(img_array)
    return embedding


# 두 얼굴간의 임베딩벡터 유클리드 공간 거리(유사도) 계산
def recognize_face(face_image, embeddings):
    face_embedding = image_to_embedding(face_image) # 현재 얼굴 이미지를 임베딩 벡터화.
    min_dist = 150
    Name = None
    # Loop over  names and encodings.
    for (name, embedding) in embeddings.items(): # 기학습된 임베딩 벡터 선형 탐색
        # 벡터 간 거리계산(기학습된 임베딩과 현재 캠 얼굴 임베딩 간
        dist = np.linalg.norm(face_embedding - embedding)
        print('%s 와의 임베딩 벡터간 거리는 - [%s] ' % (name, dist))
        if dist < min_dist:
            min_dist = dist
            Name = name
    if min_dist <= 0.75:
        return str(Name)
    else:
        return None


def recognize_faces_incam(embeddings,username,stream_url):
    count=0
    font = cv2.FONT_HERSHEY_COMPLEX
    # 라즈베리파이 카메라 실시간 영상을 받아온다.
    print("[유저의 얼굴인식 요청]")
    print("유저 이름 = " + username)
    print("요청된 유저 캠 이미지 URL ="+stream_url)
    curTime = time.time()
    fps=0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while True:
        url_response = urllib.request.urlopen("http://" + stream_url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        fps += 1
        # Loop through all the faces detected
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]
            dets = detector(face, 1)  # 얼굴 디텍팅.
            num_faces = len(dets)  # 찾은얼굴 개수
            if num_faces == 0:
                break;
            faces_list = dlib.full_object_detections()
            for detection in dets:
                faces_list.append(face_pose_predictor(face, detection))  # 68-landmark특징점을 이용한 얼굴 정렬.
            cropimage = dlib.get_face_chips(face, faces_list, size=96)
            face = cropimage[0]
            identity = recognize_face(face, embeddings)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if identity is not None: # 임베딩 벡터의 발견될때.
                cv2.rectangle(image, (x, y), (x + w, y + h), (100, 150, 150), 2)
                cv2.putText(image, str(identity).title(), (x + 5, y - 5), font, 1, (150, 100, 150), 2)
                count +=1
                print(identity+"가 인식되었습니다")
        cv2.waitKey(10)
        cv2.putText(image, str(fps), (30,20), font, 1, (200, 251, 183), 2)
        cv2.imshow('face Rec', image)
        writer.write(image)
        if count >10:
            cv2.destroyAllWindows()
            writer.release()
            return True
        if fps >240:
            cv2.destroyAllWindows()
            writer.release()
            return False



def load_embeddings():
    input_embeddings = {}
    embedding_file = np.load('embeddings.npy',allow_pickle = True)
    # embeddings.npy 에는 전에 학습시켜둔 얼굴이 저장되어있다.
    for k, v in embedding_file[()].items():
        input_embeddings[k] = v
    return input_embeddings

def init():
    with tf_session.as_default():
        with tf_graph.as_default():
            mymodel = initModel()  # 모델생성
            # mymodel = 가중치 업로드 전의 뉴럴네트웍 모델
            print("[Neural Network Model Create OK.] ")


app = Flask(__name__)
api = Api(app)
@app.route('/')
def get():
    return '<html><title>얼굴인식용 웹서버</title> <head><h1>[ 얼굴인식 웹서버 ]<br></h1></head><body><h3>서버가 정상 작동중입니다.</h3></body></html>'

#얼굴인식 결과 get, json 요청에 json으로 응답한다.
class FaceRecognize(Resource): #이 클래스는 새 쓰레드를 생성한다, 메인 쓰레드와 다르기때문에 케라스 그래프가 없다는것이다.
    K.clear_session()
    def get(self): # 결과
        with tf_session.as_default():
            with tf_graph.as_default():
                parser = reqparse.RequestParser() #요청파서 선언
                parser.add_argument('username', type=str)
                parser.add_argument('stream_url', type=str)
                args = parser.parse_args()
                _userName=args['username']
                _streamUrl = args['stream_url']
                if recognize_faces_incam(embeddings,args['username'],args['stream_url']):
                    return {'username': args['username'], 'stream_url': args['stream_url'], 'face_rec':'True'}
                else :
                    return {'username': args['username'], 'stream_url': args['stream_url'], 'face_rec': 'False'}

api.add_resource(FaceRecognize, '/facerec')

tf_session = K.get_session()  # this creates a new session since one doesn't exist already.
tf_graph = tf.get_default_graph()

if __name__=='__main__':
    with tf_session.as_default():
        with tf_graph.as_default():
            print("등록된 유저의 얼굴 데이터(embedding Vector)을 불러오는중..")
            embeddings = load_embeddings()
            print("Success")
            global facemodel
            print("얼굴인식 모델(N2.small.v2 model) 로딩중..")
            facemodel = load_model('face_model.h5')
            print("Success")
            print("얼굴인식 모델의 가중치 로드중..")
            initWeights()
            print("Success")
            print("Flask 웹서버를 실행합니다..")
            app.run(host='192.168.219.133',port=80)



