#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 13:10:38 2020

@author: baegseungho
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import initializers
# y = 3 * x1 + 5 * x2 + 10 찾아낼수있을까?
def gen_sequential_model():
    model = Sequential([
        Input(2, name='input_layer'),
        # 입력이 x1 x2 두개의 변수로부터 값을 받으니까 2, input layer의 이름은 input_layer
        Dense(16, activation='sigmoid', name='hidden_layer1', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)),
        # input layer로부터 히든레이어 구성할거임. 케라스에선 Dense라는 함수로 구성할수있음
        # 히든레이어 하나만 만들고 히든레이어안에 16개 뉴런을 쓸거고 액티베이션함수는 히든레이어에선 sigmoid쓰겠ㄱ다.
       
        Dense(1, activation='relu', name='output_layer',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))
        ])
    # input layer부터 output layer까지 시퀀스형태로 나열
    
    model.summary()
    # input 노드 2개에서 히든레이어 뉴런 16개로 화살표 다 쏘니까 16 * 2 = 32개의 가중치들. 히든레이어의 뉴런 16개 각각 바이어스값있으니까 또 16개의 파라미터 추가. 거기에 마지막 아웃풋레이어에 뉴런 하나로 16:1이니까 16개의 가중치에 바이어스값 1개해서 17개 추가.
    # 총 48개 + 17개 = 65개의 파라미터 존재
    # 이렇게 간단한 뉴럴네트와크도 파라미터의개수가 많아지는거야
    
    print(model.layers[0].get_weights())
    # 히든레이어의 가중치들
    print(model.layers[1].get_weights())
    # 아웃풋레이어의 가중치들 프린트!

    
    model.compile(optimizer='sgd', loss='mse')
    # sgd = stochastic gradient descent, mse = mean square error
    # regression이기때문에 mse쓸거고, 클래시피케이션이라면 cross entropy같은거 쓸거임..
    # 이렇게 옵티마이저랑 로스함수 지정해주고 돌리게됨
    return model


# y = w1 * x1 + w2 + x2 + b
def gen_linear_regression_dataset(numofsamples=500, w1 = 3, w2 = 5, b = 10): 
# 실험을 위한 데이터셋 y만들기
    np.random.seed(42)
    X = np.random.rand(numofsamples, 2)
    #0 ~ 1사이의 값을 줌. 샘플의 개수, n차원 어레이. x1과 x2값을 만들어야하니까..2차원으로 x,y 한쌍씩 아무거나 생성하면됨
    coef = np.array([w1, w2])
    # 함수인자로 주는 w1 w2를 계수로보고 b를 bias로 봄
    bias = b
    y = np.matmul(X, coef.transpose()) + bias
    # for loop안돌기위해 벡터연산! matmul()
    return X, y

# 학습 history를 플랏으로 그려보자
def plot_loss_curve(history):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,10))
    # 그래프 크기 15*10
    plt.plot(history.history['loss'][1:])
    # training data에대한 loss
    plt.plot(history.history['val_loss'][1:])
    # validation data에대한 loss
    plt.title('model loss')
    plt.ylabel('loss')
    # loss어케되는지보자
    plt.xlabel('epoch')
    # epoch쭉 증가
    plt.legend(['train','test'], loc='upper right')
    plt.show()
    
# 만들어진 모델이용해서 새로운 데이터셋 만든거 넣어보자!
def predict_new_sample(model, x, w1=3, w2=5, b = 10):
    x = x.reshape(1,2)
    y_pred = model.predict(x)[0][0]
    y_actual = w1*x[0][0] + w2*x[0][1] + b
    
    print("y actual value = ", y_actual)
    print("y predicted value = ", y_pred)
    



model = gen_sequential_model()
# 모델만들고
X, y = gen_linear_regression_dataset(numofsamples=1000)
# 트레이닝데이터셋 만들고
history = model.fit(X, y, epochs=100, verbose=2, validation_split=0.3)
# 30번 반복. 실제 트레이닝 각 단계에서의 로스값, 주어진 데이터 자체적으로 train과 validataion스플릿할수있게끔 validation_split=0.3 => training에 70퍼 validate 하는데 30퍼
plot_loss_curve(history)

# weight값이 초기화가 잘못된값이 들어가면 로스값 엄청커져..
# 로컬미니마에 빠지는경우가 있음..
# 어떻게 피하지? 매번돌릴때마다 값이 달라
# 패러미터를 초기화를 고정시켜주는거. 아까 파라미터 65개라그랬음
# Dense에서 kernel_initializer 라는걸 사용하면됨.
# Dense(1, activation='relu', name='output_layer',kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05), seed=42)

print("train loss=", history.history['loss'][-1])
print("test loss=", history.history['val_loss'][-1])
# 마지막 로스값 출력(train, test)
# 데이터 개수가 많으면 만들수록 로스값 적어지넹


predict_new_sample(model, np.array([0.1, 0.2]))


# 뉴럴네트워크는 어떤 머신러닝 문제든 풀수있어
# 데이터 충분히주면 학습을 하기때문
