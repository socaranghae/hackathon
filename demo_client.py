import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import torch
from torchvision import transforms
from PIL import Image


def get_lables(data, col):
    '''
    주어진 data frame(data)에서 입력된 열(col)의 unique 값을 모아
    index로 호출 가능한 dictionary를 만들어 반환한다.
    '''
    labels = sorted(list(data[col].unique()))
    label_dict = {i : key for i, key in enumerate(labels)}
    return label_dict

def load_mood_image(file):
    '''
    업로드된 이미지를 보여주고, model에 넣을 수 있는 형태로 변환하여 반환한다.
    '''
    img = Image.open(file)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img_array = np.array(img)
    img_array = img_array/255.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_place_image(file):
    img = Image.open(file)
    transformer = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        ])
    img_array = transformer(img)
    img_array = img_array.expand(1, -1, -1, -1)
    return img_array
    
def get_coupon(data, mood, place):
    '''
    사진에서 추출된 mood와 place에 따라 지정된 맞춤 쿠폰을 발행한다.
    '''
    pick_coupon = data[data['mood'] == mood][data['place']==place].reset_index(drop=True)
    name = pick_coupon.loc[0, 'name']
    coupon = pick_coupon.loc[0, 'coupon']
    st.markdown(name)
    st.markdown(f'쏘카러버를 위한 맞춤 쿠폰 [{coupon}]이 1장 발행되었습니다.')


def main():
    '''
    main 작동 함수
    '''
    col1, col2 = st.columns([1,6])
    col1.image('SOCAR_Pin_RGB_블루.png')
    col2.title("차가 필요한 모든 순간, 쏘카")
    col2.subheader("쏘카와 함께한 오늘의 여정은 어떠셨나요?")

    # load the coupon list and trained models with label lists
    coupon_list = pd.read_csv('coupon_list.csv')
    mood_model = keras.models.load_model('mood_classifier_dw.h5')
    mood_label = get_lables(coupon_list, 'mood')

    device = torch.device('cpu')
    place_model = torch.load('model_final.pth', map_location=device)
    place_model.eval().to(device)
    place_label = get_lables(coupon_list, 'place')

    # Upload picture
    pic = st.file_uploader(label='', type=['png', 'jpg', 'jpeg'])
    if pic:
        st.image(pic)
        mood_img = load_mood_image(pic)
        place_img = load_place_image(pic)

        mood_i = mood_model.predict(mood_img)
        mood_i = np.argmax(mood_i)
        mood = mood_label[mood_i]

        place_i = place_model(place_img)
        _, place_i = torch.max(place_i, 1)
        place_i = int(place_i)
        place = place_label[place_i]

        get_coupon(coupon_list, mood, place)


if __name__ == '__main__':
    main()
