import numpy as np
from tensorflow.python.keras.layers import Input
import model
from model import create_model
import utils
from create_embedding import create_input_image_embeddings


model = create_model(Input)
# 기학습된 모델 가중치 불러오기.
weights = utils.weights
weights_dict = utils.load_weights()
# Set layer weights of the model
for name in weights:
    print("[+] Setting weights........... ")
    if model.get_layer(name) != None:
        model.get_layer(name).set_weights(weights_dict[name])
    elif model.get_layer(name) != None:
        model.get_layer(name).set_weights(weights_dict[name])
input_embeddings = create_input_image_embeddings()
print('임베딩 벡터를 저장합니다')
#임베딩 벡터 저장
np.save("embeddings.npy", input_embeddings)
print('임베딩 벡터 저장 완료')