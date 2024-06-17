#########오리지널 설명자료 ###### https://www.youtube.com/watch?v=PwYnZ2RwyJw 
# import tensorflow as tf

# 키 = 170
# 신발 = 260
# # 신발 = a x 키 + b

# # 실제값 = 260
# # 예측값 = a x 실제값 + b

# a = tf.Variable(0.1)  # tf.Variable 은 W 값만드는 문법
# b = tf.Variable(0.2)  # 0.1, 0.2 는 일단 임으로 정하면 됨 

# def 손실함수():             # 손실함수 = 오류(오차)값
#     신발예측값 = a * 키 + b
#  #   실제값 = 260
#     return tf.square(260 - 신발예측값)

# opt = tf.keras.optimizers.Adam(learning_rate=0.1)   # a,b를 경사하강법으로 구하기 

# for i in range(300):
#     opt.minimize(손실함수, var_list=[a,b])    # 경사하강 1번 해줌 == a,b를 한번 수정해줌
#     print(a.numpy(),b.numpy())  # print(a,b)  에서 값만 따로 출력하려면 , numpy
#########################################################################################

# tf.keras.optimizers.Adam 클래스는 이제 minimize 메서드를 지원하지 않음.
# 즉, Adam이 지원안함.대신 tf.GradientTape와 apply_gradients를 사용
#

import tensorflow as tf

키 = 170
신발 = 260
# 신발 = a x 키 + b

# 실제값 = 260
# 예측값 = a x 실제값 + b

a = tf.Variable(0.1)  # tf.Variable 은 W 값만드는 문법
b = tf.Variable(0.2)  # 0.1, 0.2 는 일단 임으로 정하면 됨 

def 손실함수():  # 손실함수 = 오류(오차)값
    예측값 = a * 키 + b
    return tf.square(260 - 예측값)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)  # a,b를 경사하강법으로 구하기 

# tf.GradientTape을 사용하여 변수를 업데이트하는 방식으로 수정
for step in range(100):  # 100번의 스텝을 실행합니다.
    with tf.GradientTape() as tape:
        loss = 손실함수()
    gradients = tape.gradient(loss, [a, b])
    opt.apply_gradients(zip(gradients, [a, b]))

print(f"a: {a.numpy()}, b: {b.numpy()}")

def 신발사이즈(a, b):
    키 = float(input("키가 몇이죠? "))
    신발예측 = a * 키 + b
    print(f"예측 신발 사이즈: {신발예측}")

# 학습된 a와 b를 사용하여 신발 사이즈 예측
신발사이즈(a, b)

