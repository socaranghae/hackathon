# Like LION X SOCAR Hackathon project


#### 팀명 : 쏘카랑해
#### 프로젝트 명 : 이미지 인식을 통한 개인화 커뮤니케이션 & 쿠폰 발급 모델 
#### 기획 배경/목적 : 유저가 주행을 완료한 후 업로드한 사진을 분석하여 개인화된 커뮤니케이션을 시행하고 쿠폰 지급
#### 개발 범위 및 링크
* Place Predict Model (Place Classifier)
  - Place 데이터 수집을 위한 구글 크롤링 코드 개발 
    + [GoogleCrawling.ipynb](https://github.com/socaranghae/hackathon/blob/main/GoogleCrawling.ipynb)
    + [구글 이미지 크롤러_수진.ipynb](https://github.com/socaranghae/hackathon/blob/main/%EA%B5%AC%EA%B8%80%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%ED%81%AC%EB%A1%A4%EB%9F%AC_%EC%88%98%EC%A7%84.ipynb)
  - Place Predict Model 개발 및 Train (각각 다른 Pretrained model을 활용하여 Train)
    + [Inception_v3_1.ipynb](https://github.com/socaranghae/hackathon/blob/main/Inception_v3_1.ipynb)
    + [Inception_v3_2.ipynb](https://github.com/socaranghae/hackathon/blob/main/Inception_v3_2.ipynb)
    + [PJT_placepredict_3.ipynb](https://github.com/socaranghae/hackathon/blob/main/PJT_placepredict_3.ipynb)
    + [place_model_wMobileNet.ipynb](https://github.com/socaranghae/hackathon/blob/main/place_model_wMobileNet.ipynb)
  - 최종 모델 : train 결과 가장 높은 정확도를 나타낸 모델을 사용하기로 결정 
    + [PJT_placepredict_4.ipynb](https://github.com/socaranghae/hackathon/blob/main/PJT_placepredict_4.ipynb)    
* Mood Predict Model (Mood Classifier)
  - Emotion6 Dataset 활용
  - 최종 모델 : [_Mood_Classifier_ResNet50.ipynb](https://github.com/socaranghae/hackathon/blob/main/_Mood_Classifier_ResNet50.ipynb)
* Demo Client(web) : 각 Model의 결과를 바탕으로 매칭되는 쿠폰 발급
  - [demo_client.py](https://github.com/socaranghae/hackathon/blob/main/demo_client.py)
