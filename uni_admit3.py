# 1. import후, 데이터 불러오기
# 2. X,y 데이타로 분리하기
# 3. 딥러닝 모델 디자인하고, 컴파일하기
# 4. 모델 fit 하기
# 5. 예측함수 정의하기
###############################################################################

import pandas as pd
import numpy as np
import tensorflow as tf
#1
df = pd.read_csv('D:/dev_D/DeepLearning/input/gpascore.csv')
print(df.head())
print(df[:10])
print(df.isnull().sum())
# exit()
df.fillna(df.mean(), inplace = True)
print(df)
# exit()
print(df.isnull().sum())
#2
X_data = df[['gre', 'gpa', 'rank']].values
y_data = df['admit'].values
#3
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation = 'tanh'),
    tf.keras.layers.Dense(64, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),
])
model.compile(optimizer='adam',loss = 'binary_crossentropy', metrics = 'accuracy')

#4
model.fit(X_data, y_data, epochs = 300)

#5 
def 대입예측():
    try:
        gre = int(input("GRE성적은?"))
        gpa = float(input("GPA성적은?"))
        rank = int(input("대학 랭킹은?"))
        입력값 = np.array([[gre,gpa,rank]])
        예측값 = model.predict(입력값)
        print(f"귀하의 대학 합격 가능성은 {예측값} 입니다.")
    except ValueError:
        print("올바른 형식으로 입력해 주세요!")
대입예측()
