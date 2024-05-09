# 1. import, df=read_csv., print(df.head()), print(df.shape), 
# print(df.isnull().sum()), print(df.columns),df.fillna(df.mean(),inplace = True)
# 2. X_data, y_data 분리
# 3. tf 모델 디자인, 컴파일 수행 
#   - model = tf.keras.models.Sequential([
#           tf.keras.layers.Dense()
# 4. model.fit(X,y,epochs=3)
# 5. def 대입예측():
#       try: , except ValueError:
# 대입예측()
##################################################
#1. 데이터 읽기와 분석
import pandas as pd
import numpy as np
import tensorflow as tf
df = pd.read_csv('D:/dev_D/DeepLearning/input/gpascore.csv')
# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)
print(df.isnull().sum())

# 2. X,y데이타 분리
X_data = df[['gre','gpa','rank']].values
y = df['admit'].values

# 3. 모델 디자인과 컴파일
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128,activation = 'tanh'),
    tf.keras.layers.Dense(64, activation = 'tanh'),
    tf.keras.layers.Dense(24, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),

])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy' )

# 4. 모델훈련

model.fit(X_data,y, epochs = 300)

# 5. 예측 함수 생성

def 예측():
    try:
        gre = int(input("gre몇점?"))
        gpa = float(input("gpa몇점?"))
        rank = int(input("rank몇등급?"))

        prediction = model.predict(np.array([[gre,gpa,rank]]))
        print(f"귀하의 대입 합격 가능성은 {prediction} 입니다.")


    except ValueError:
        print(f"올바른 숫자양식으로 입력하시요 !!")
예측()