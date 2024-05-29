
################################################################################
# 1. 데이터 불러오기
# 2. X데이터와 y데이터 분리
# 3. 딥러닝 모델 디자인 & 컴파일 하기 '''
''' model = tf.keras.models.Sequential([
    레이어1(),
    레이어2(),
    레이어3(), sigmoid 는 모든값을 0~1 사이의 값으로 압축시켜주는 activation 함수
]) 
    model.compile  까지가 모델 디자인 단계임
'''
# 4. 모델 학습(Fit) 시키기 
''' model.fit(X,y, epochs = 100)'''

# 5. 예측함수 정의하기
######################################################################
import pandas as pd
import numpy as np
import tensorflow as tf

# 1.데이터 불러오기
data = pd.read_csv('D:/dev_D/DeepLearning/input/gpascore.csv')
data.fillna(data.mean(), inplace = True)
''' 데이터 검토 명령어들... '''
# df = pd.read_csv('D:/dev_D/DeepLearning/input/gpascore.csv')
# print(df.head())
# print(df[:10]) 첨부터 10개
# print(df[-1:])  끝에서 1개
# print(df.shape) 행열 갯수 출력
# print(df.columns) , 컬럼명 출력
# print(df.isnull().sum()), 컬럼값중에 null인 행의 갯수
# data[data.isnull().any(axis=1)] 컬럼 값에 null이 있는 행을 출력
# # exit()
# df.fillna(df.mean(), inplace = True)
# print(df)
# # exit()
# print(df.isnull().sum())

# 2.X데이터와 y데이터 분리
y데이타 = data['admit'].values
X데이타 = data[['gre','gpa','rank']].values

# 3.모델 디자인하기
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128,activation= 'tanh'),
    tf.keras.layers.Dense(64,activation = 'tanh'),
    tf.keras.layers.Dense(1,activation = 'sigmoid'),
])

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# 4. 모델 훈련
model.fit(X데이타, y데이타, epochs = 300)

# 5.예측함수 정의
def 대입합격():
    try:
        a = int(input("GRE?"))
        b = float(input("GPA?"))
        c = int(input("대학랭킹?"))

        입력값 = np.array([[a,b,c]])
        예측값 = model.predict(입력값)
        print(f"귀하의 대학 합격여부 퍼센티지는 {예측값} 입니다.")
    except ValueError:
        print("올바른 형식으로 입력해 주세요!!")
대입합격()



