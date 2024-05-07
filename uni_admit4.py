'''  대입 합격 예측 모델
1. import pandas,numpy,tensorflow 후 데이타 불러오기
2. 데이터 X,y값으로 분리하기
3. 딥러닝 모델 과 컴파일 만들기
4. 모델 fit  하기
5. 예측 함수 만들기 
'''
import numpy as np
import tensorflow as tf
import pandas as pd

df = pd.read_csv('D:/dev_D/DeepLearning/input/gpascore.csv')
# print(df.head())
print(df.isnull().sum())
# exit()
df.fillna(df.mean(), inplace = True)
# exit()
X_data = df[['gre','gpa','rank']].values
y_data = df['admit'].values

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation = 'tanh'),
    tf.keras.layers.Dense(64, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),
])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_data, y_data, epochs = 300)

def 대입예측():
    try:
        gre=int(input("gre값은?"))
        gpa=float(input("gpa값은?"))
        rank=int(input("대학rank는?"))
        
        입력값 = np.array([[gre,gpa,rank]])
        예측값 = model.predict(입력값)
        print(f"귀하의 대입 합격예측값은 {예측값} 입니다.")
    except ValueError:
        print("올바른 값을 넣으시요!!")
대입예측()

    