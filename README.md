#[face recognition](https://github.com/jaehyunup/face_recognition_RestAPI)
-----
#### Directory
- face_recognizer
  - webserver.py - 웹서버 동작 및 faceRecognition 요청에 관한 전반적인 처리를 모두 하고있음
  - utils.py - 얼굴인식 모델 생성 및 별개의 기능들을 미리 정의한 파일

<br>

### 개발 계기
----
저번학기 프로젝트 진행중 스마트 홈 시스템을 프로젝트로 하여 내부적으로 얼굴인식을 통한 보안기능을 넣었었다.

하지만 스마트홈 특성상 MCU 기기(프로젝트에서는 Raspberry Pi 3 B+ Model) 를 사용하는 환경이 대부분이고 이런 환경구성에서 Image Processing과 머신러닝을 사용한다는 것은 불가능 했다
왜냐..**너무 느려!**

혹시나 하는 마음에 OpenFace에서 제공하는 얼굴인식 모듈을 어찌어찌 rasbian에 설치했다.
실행하니 한 두개의 이미지만 30분 비교하다가 메모리 부족으로 에러가 뜬다..에효

그래서 이 OpenFace를 활용한 얼굴인식 부분만 REST API로 개발해서 GTX 1060이 달린 내 개인 PC에 연산을 맡길것이다. 

카카오나 네이버에서 제공하는 RESTAPI의 그것과 같은 서비스들을 만들어보고싶다고 항상 생각했었는데 이참에 REST API도 만들어보고 마이크로서비스가 어떻게 구성되어지는가에 대한 이해도 겸할수 있을것 같아 좋은 경험일것 같았다.

<br>

혹여나 일반적인 리눅스환경에서 OpenFace모듈을 사용하고싶다면 
[OpenFace로 우리 오빠들 얼굴 인식하기](https://www.popit.kr/openface-exo-member-face-recognition/) 위 링크를 보길 바란다.
이분이 OpenFace의 디렉토리 구조까지 아주 잘 설명해뒀다.

<br>

---

### OpenFace

[Openface](https://cmusatyalab.github.io/openface/)는 얼굴 유사도측정 오픈소스이다.




## OVERVIEW
> 1. Detect faces with a pre-trained models from dlib or OpenCV.
> 
> 2. Transform the face for the neural network. This repository uses dlib's real-time pose estimation with OpenCV's affine transformation to try to make the eyes and bottom lip appear in the same location on each image.<br>
> 
> 3. Use a deep neural network to represent (or embed) the face on a 128-dimensional unit hypersphere. The embedding is a generic representation for anybody's face. Unlike other face representations, this embedding has the nice property that a larger distance between two face embeddings means that the faces are likely not of the same person. This property makes clustering, similarity detection, and classification tasks easier than other face recognition techniques where the Euclidean distance between features is not meaningful.

<br><br><br>

--- 

## 아이디어
 openface의 기학습된 딥러닝 모델에 얼굴 정규화과정만 거쳐서 predict한 결과는 
128-embedding Vector로 표현되어지고 그 결과는 그 사람만의 128-embedding Vector인것이다.

즉, 얼굴이 완전 다르게 생긴 철수와 영희의 얼굴사진을 Openface의 딥러닝 모델에 넣고 predict한 결과인 철수의 embedding Vector와 영희의 embedding Vector는 멀리 떨어져 표현된다는 것이다.

반대로 얼굴이 비슷하게 생긴 철수와 철구의 얼굴사진을 넣어 predict한 결과로 얻은 두개의 128-embedding vector는 유클리드 공간상의 거리가 가깝다!(얼굴이 비슷하니까)


이제 우린 OpenFace가 어떤식으로 얼굴의 유사도 측정을 하는지 알게 되었다.



-----
### OpenFace 사용하기

<img src="/img/openface_artifact.png" width="600" style="display:block;margin-left:auto; margin-right:auto;margin-top:20px;">


<br>
Openface는 Torch 기반으로 학습시킨 기학습된 Neural NetWorkModel을 제공중이다.
그것이 그림에서 보라색 박스로 표현된 부분이고, 결과적으로 학습된 뉴럴네트웍을 제공해주는데 사용자가 사용하기 위해서는, 위에서 말한 정규화 Processing을 거치고 기학습되어 제공되는 NeuralNetwork에 Predict한 결과로 Classification을 하여 얼굴 정확도를 검증하는것이다.

그렇다면 위에서 우리가 구현해야할 부분은 빨간색 박스 부분이다.

저부분을 구현해서, 사용자의 Embedding Vector를 저장하고, 얼굴 인식 요청이 왔을때
현재 카메라에 감지되는 사람과 저장된 Embedding Vector 리스트를 Classification  **저장된 임베딩벡터와의 유사도가 아주높다고 판단되었을때만** True를 반환해주는 REST API로 구현할 것이다.





-----
### REST API 소개
위에서 설명한 것을 토대로 REST API를 개발했는데 사용자가 이용하는 유스케이스로 간단하게 표현한 구조는 아래와 같다.

<img src="/img/rest_artifact.png" width="800" style="display:block;margin-left:auto; margin-right:auto;margin-top:20px;">

##### 작동순서
1. 사용자는 REST API에 카메라 영상을 포함하여 얼굴인식요청을 한다
2. 요청을 받은 서버는 카메라 영상에서 얼굴 영역을 인식한다(Haar Cascade 이용) 
3. 인식된 얼굴영역을 자른다
4. Dlib을 이용해 얼굴의 68-landmark을 구분하고 68-landmark중 코에 해당하는 Landmark부분을 중앙으로 오게하여 이미지의 중앙에 코가 올수있도록 얼굴 위치를 이동시킨다.
5. 이 이미지를 OpenFace에서 기학습한 Neural Network Model을 이용해 Predict값을 받는다
6. Embedding Vector를 통해 얼굴 유사도를 측정한다 

<br>

-----

### 구현결과

<img src="/img/parkjaehyun.png" width="150" style="display:block;margin-left:auto; margin-right:auto;margin-top:20px;">
<img src="/img/embedding.gif" width="400" style="display:block;margin-left:auto; margin-right:auto;margin-top:20px;">
<br>
요청이 들어왔을때 RESTAPI를 가동중인 서버는 임베딩 벡터간의 거리를 측정하고, (일반적으로 0.6~7정도 아래로 내려가면 거의 같은사람이라고 보면 된다고 한다)
해당되는 사람이 발견된다면 위와같이 Json형태로 결과를 반환해준다.


