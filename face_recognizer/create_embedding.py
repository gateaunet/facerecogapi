import cv2
import glob
import os
import numpy as np
from tensorflow.python.keras.layers import Input
import model
from model import create_model
import utils
Input = Input(shape=(96, 96, 3))
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# 학습용 이미지를 임베딩으로 변환하는 과정
def image_to_embedding(image, model):
    image = cv2.resize(image, (96, 96))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
    img_array = np.array([img])
    # 기학습된 모델에 이미지를 입력으로 이미지에 대한 embedding vector를 반환한다.
    embeddings = model.predict_on_batch(img_array)
    return embeddings

# 학습 디렉토리에있는 이미지를 128-D 임베딩 벡터로 변환하여 inpurt_embeddings 객체 반환
def create_input_image_embeddings():
    input_embeddings = {}
    for file in glob.glob("train_images/*"):
        #확장명 없는 순수 파일명(이름)가져오기.
        person_name = os.path.splitext(os.path.basename(file))[0]
        image = cv2.imread(file, 1)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.2, 5)
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]
            input_embeddings[person_name] = image_to_embedding(face, model)

    return input_embeddings


model = create_model(Input)
# 기학습된 모델 가중치 불러오기.
weights = utils.weights
weights_dict = utils.load_weights()

# Set layer weights of the model
for name in weights:
    print("가중치를 셋팅중입니다.")
    if model.get_layer(name) != None:
        model.get_layer(name).set_weights(weights_dict[name])
    elif model.get_layer(name) != None:
        model.get_layer(name).set_weights(weights_dict[name])
input_embeddings = create_input_image_embeddings()
print('임베딩 벡터를 저장합니다')
#임베딩 벡터 저장
np.save("embeddings.npy", input_embeddings)
print('임베딩 벡터가 저장되었습니다')