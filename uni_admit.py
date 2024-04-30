
''' 1단계 딥러닝 모델 디자인하기 '''
''' model = tf.keras.models.Sequential([
    레이어1(),
    레이어2(),
    레이어3(), sigmoid 는 모든값을 0~1 사이의 값으로 압축시켜주는 activation 함수
]) 
    model.compile  까지가 모델 디자인 단계임
'''
# # 2단계 모델 학습(Fit) 시키기 
''' model.fit(X,y, epochs = 1000)'''
######################################################################

import pandas as pd
data = pd.read_csv('D:/dev_D/DeepLearning/input/gpascore.csv')
# print(data)
# print(data.isnull().sum())
data = data.dropna()
# print(data['gre'].count())
# exit()

''' [ [380,3.21,3], [660,3.67.3], [800,4,1],[],[] ...] gre,gpa,rank를 리스트로 정리하고
    [ 0, 1, 1 ... ]   admit 도 리스트로(어레이타입)으로 정리해야함. '''
# 이것을 pandas로 코딩하면...
y데이타 = data['admit'].values
X데이타 = []
for i, rows in data.iterrows():
    X데이타.append([rows['gre'], rows['gpa'], rows['rank']])
print(X데이타)
# exit()

import numpy as np  # model.fit에서 X데이타값을 어레이 타입으로 받아들이기 위해 사용 
import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation ='tanh'), # 숫자는 노드 갯수:임의선정가능
    tf.keras.layers.Dense(128, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),   # 마지막 노드갯수를 1로 해야함.
]) 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# binary_crossentropy 함수는 결과가 0과1사이의 분류/확률문제에 사용

model.fit(np.array(X데이타), np.array(y데이타), epochs = 100)

# 예측
###### 수정이 필요한 초기 코드 ###############################
# def 대입합격(a,b,c):
#     예측값 = model.predict([[a, b, c]])  >>> 이상함
# a = int(input("GRE성적?"))
# b = float(input("GPA성적?"))
# c = int(input("대학랭킹?"))

# 예측값 = model.predict([[a,b,c]])
# print(f"귀하여 대입 합격여부는 {예측값} 입니다.")	

# 대입합격(a,b,c)
################## GPT 이용해서 수정 완료한 코드 ###################################
# 예측
def 대입합격():

    try:
        a = int(input("GRE 성적? "))
        b = float(input("GPA 성적? "))
        c = int(input("대학 랭킹? "))

        입력값 = [[a, b, c]]
        예측값 = model.predict(입력값) # 윗/아래 입력값 없애고 대신 ([[a,b,c]]) 써도됨
        print(f"귀하의 대입 합격 여부는 {예측값} 입니다.")  # 예측값 출력

    # 예측값 = model.predict([[a,b,c]]),원래는 a,b,c 대신 400, 3.7, 3 으로 들어감.
    # print(f"귀하의 대입 합격여부는 {예측값} 입니다.")	~~~~~~


    except ValueError:
        print("올바른 형식으로 입력해주세요.")

대입합격()

#############################################################
# ## 또는 직접 값을 입력한것을 수행하는 방식이 있음

# 예측값 = model.predict([[750,3.70,3], [400,2.2,1]])
# print(f"귀하여 대입 합격여부는 {예측값} 입니다.")	
#############################################################

exit()

#####################정상 작동하지만 이상한 코드#############################################################

def 대입합격(a, b, c):
    입력값 = [[a, b, c]]
    예측값 = model.predict(입력값)
    print(f"귀하의 대입 합격 여부는 {예측값} 입니다.")  # 예측값 출력

a = int(input("GRE 성적? "))
b = float(input("GPA 성적? "))
c = int(input("대학 랭킹? "))

대입합격(a, b, c)

exit()
##############################################
def 대입합격(a,b,c):
    입력값 = [[ ]] #model.predict([[a, b, c]])
a = int(input("GRE성적?"))
b = float(input("GPA성적?"))
c = int(input("대학랭킹?"))

예측값 = model.predict([[a,b,c]])
print(f"귀하의 대입 합격여부는 {예측값} 입니다.")	

대입합격(a,b,c)




