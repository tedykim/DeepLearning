# 1.import하고 데이터 읽고, 데이터 분석하기

import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv('D:/dev_D/DeepLearning/input/gpascore.csv')
# print(df.head())
# print(df.shape)
# print(df.isnull().sum())
# print(df.columns)
df.fillna(df.mean(),inplace = True)
# print(df.isnull().sum())

# 2.X,y 데이타 분리지정
X_data = df[['gre','gpa','rank']].values
y_data = df['admit'].values

# 3.모델 디자인하고, 모델을 컴파일 설정

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128,activation = 'tanh'),
    tf.keras.layers.Dense(64,activation = 'tanh'),
    tf.keras.layers.Dense(24,activation = 'tanh'),
    tf.keras.layers.Dense(1,activation = 'sigmoid'),
])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy' )

# 4.모델 fit

model.fit(X_data, y_data, epochs = 300)

# 5.예측함수 정의
def 예측():
    try:
        gre = int(input("gre몇점일까요?"))
        gpa = float(input("gpa 몇점일까요?"))
        rank = int(input("rank 순위는?"))

        prediction = model.predict(np.array([[gre,gpa,rank]]))
        print(f"귀하의 대입합격 예측율은 {prediction} 입니다.")

    except ValueError:
        print("적정값을 입력하세요..")

예측()