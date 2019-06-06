import cv2
import glob
import os
import numpy as np
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.python.keras.layers.core import Lambda, Flatten, Dense
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.pooling import MaxPooling2D, AveragePooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

import create_embedding
import utils
import urllib
face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def weightinit(model):
    weights = utils.weights
    weights_dict = utils.load_weights()
    for name in weights:
        if model.get_layer(name) != None:
            model.get_layer(name).set_weights(weights_dict[name])
        elif model.get_layer(name) != None:
            model.get_layer(name).set_weights(weights_dict[name])


# small2.nn2 모델 생성 및 기학습된 가중치 초기화.

# 현재 얼굴 이미지를 임베딩 벡터화 시킴.
def image_to_embedding(image, model):
    image = cv2.resize(image, (96, 96))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
    img_array = np.array([img])
    embedding = model.predict_on_batch(img_array)
    return embedding


# 두 얼굴간의 임베딩벡터 유클리드 공간 거리(유사도) 계산
def recognize_face(face_image, embeddings, model):
    face_embedding = image_to_embedding(face_image, model) # 현재 얼굴 이미지를 임베딩 벡터화.
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


def recognize_faces_incam(embeddings,username,stream_url,model,sess):
    K.set_session(sess)
    count=0
    font = cv2.FONT_HERSHEY_COMPLEX
    # 라즈베리파이 카메라 실시간 영상을 받아온다.
    print("웹캠의 요청 URL ="+stream_url)
    while True:
        url_response = urllib.request.urlopen("http://" + stream_url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        # Loop through all the faces detected
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]
            identity = recognize_face(face, embeddings, model)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            print("얼굴검출")
            if identity is not None: # 일치하는 임베딩벡터의 이름 발견될때.
                cv2.rectangle(image, (x, y), (x + w, y + h), (100, 150, 150), 2)
                cv2.putText(image, str(identity).title(), (x + 5, y - 5), font, 1, (150, 100, 150), 2)
                print(username+" 의 얼굴이 인식되었습니다.")
                if identity==username:
                    count +=1
        cv2.waitKey(10)
        cv2.imshow('face Rec', image)
        if count >10:
            return True



def load_embeddings():
    input_embeddings = {}
    embedding_file = np.load('embeddings.npy',allow_pickle = True)
    # embeddings.npy 에는 전에 학습시켜둔 얼굴이 저장되어있다.
    for k, v in embedding_file[()].items():
        input_embeddings[k] = v
    return input_embeddings

#embeddings = load_embeddings() # 웹서버 실행시 임베딩을 불러옴으로서, 시간단축
#recognize_faces_incam(embeddings)


