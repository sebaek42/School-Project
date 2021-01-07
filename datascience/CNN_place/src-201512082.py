#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 21:20:09 2020

@author: baegseungho
"""
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#https://youtu.be/J6VSiqYBzi4
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def set_dataframe():
    filenames = os.listdir('./images2')
    category = []

    # flow_from_dataframe 사용하기 위해 데이터프레임으로 한 이미지당 (파일명, 속한카테고리)로 분류해준다.
    for filename in filenames:
        
        
        if 'food' in filename:
            category.append(0)
        elif 'interior' in filename:
            category.append(1)
        else:
            category.append(2)

    df = pd.DataFrame({
        'filename': filenames,
        'category': category
    })
    
    return df
# BatchNormalization gradient vanishing이나 Exploding이 발생하지 않는 안정적이고 빠른 학습. Dropout과적합방지용. 첨엔 적게 갈수록 많이 . padding은 경계처리방법. 유효한영역만사용...속도 높임
# 영상을 일차원으로 바꿔주는 Flatten..2차원자료를 다루지만 전결합층에 전달하기위해선 1차원자료로 바꿔줘야함
def train_model(df):
    model = Sequential([
        Input(shape=(300,300,3), name='input_layer'),
        Conv2D(32, (3, 3), padding='valid', activation='relu', name='conv2d_layer1'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), padding='valid', activation='relu', name='conv2d_layer2'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.33),
        Conv2D(128, (3, 3), padding='valid', activation='relu', name='conv2d_layer3'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.40),
        Flatten(),
        Dense(48, activation='relu', name='fully_connected_layer'),
        BatchNormalization(),
        Dropout(0.50),
        Dense(3, activation='softmax', name='output_layer')
    ])

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #오버피팅 방지..3회까진 봐줌
    es = EarlyStopping(patience=3)
    # validate accuracy를 모니터링해서 로컬미니마에 빠진경우 학습율을 1/2로 줄여줌으로써 빠져나오도록 유도. 다만 학습률의 마지노선은 0.0005로 설정
    # verbose=1, 언제 학습멈췄는지 화면에 출력. patience는 성능증가하지 않은 에포크 몇번이나 허용할것인가.
    lr = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
    #너무 많은 에포크는 오버피팅을 일으킨다. 하지만 너무 적은 에포크는 언더피팅을 일으킨다.
    #이 상황에서 에포크 어떻게 설정해야하는ㄱ..무조건 많이 돌리도록 설정하고 특정 시점에서 멈추도록함.. 
    
    
    checkpoint_filepath = '/Users/baegseungho/건국대학교/2020-2/데이터사이언스'
    cp = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    callbacks = [es, cp, lr]
    
    df['category'] = df['category'].replace({0: '1food', 1: '2interior', 2:'3exterior'})  
    #데이터프레임을 train용과 validation용으로 나눔. 20퍼만큼은 테스트로 사용. 랜덤스테이트42로 셔플링해서 적절히 섞일수있도록함.
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_length = train_df.shape[0]
    val_length = val_df.shape[0]


    #학습용 배치파일생성시 정규화 밑 데이터 전처리
    train_datagen = ImageDataGenerator(
    #rotation_range=10,
    #rescale=1./255,
    #shear_range=0.1,
    #zoom_range=0.2,
    #horizontal_flip=True,
    #width_shift_range=0.1,
    #height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        "./images2", 
        x_col='filename',
        y_col='category',
        target_size=(300,300),
        class_mode='categorical',
        batch_size=16
    )

    #Validation Generator 얘는 검증용이니까 다른건 건들지말고 스케일만 낮추자
    validation_datagen = ImageDataGenerator(
        #rescale=1./255
        )
    #validation용 데이터프레임으로부터 넘파이어레이형태 데이터 추출. 데이터프레임으로부터 (배치사이즈, (이미지크기), 채널수) 넘파이어레이 전처리가능. classmode categorical로 onehot인코딩된 라벨이 반환됨
    validation_generator = validation_datagen.flow_from_dataframe(
        val_df, 
        "./images2", 
        x_col='filename',
        y_col='category',
        target_size=(300,300),
        class_mode='categorical',
        batch_size=16
    )


    # 몇번 반복학습할지 에포크 결정하고 model.fit호출로 학습시작
    # Traditionally, the steps per epoch is calculated as train_length // batch_size, since this will use all of the data points, one batch size worth at a time
    history = model.fit(
        train_generator, 
        epochs=8,
        validation_data=validation_generator,
        steps_per_epoch=train_length//16,
        validation_steps=val_length//16,
        callbacks=callbacks
    )
    print(history.history.keys())
    plot_loss_curve(history.history)
    model.save('model-201512082')



    
def plot_loss_curve(history):
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    
    loss_ax.plot(history['loss'], 'y', label='train loss')
    loss_ax.plot(history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')
    acc_ax.plot(history['accuracy'], 'b', label='train acc')
    acc_ax.plot(history['val_accuracy'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper right')
    
    plt.show()
    
def test_model(model):
    example_generator = ImageDataGenerator().flow_from_directory(
        'examples',
        target_size = (300,300),
        batch_size=1,
        shuffle=False
        )
    true_class = []
    for i in range(0,30):
        x_train, y_train = example_generator.next()
        true_class.append(y_train)
    predict_class = model.predict(example_generator, steps=30)
    print(true_class)
    print(example_generator.class_indices)
    print(predict_class)

    
    
    
if __name__ == '__main__':
    df = set_dataframe()
    #train_model(df)
    model = load_model('model-201512082')
    test_model(model)