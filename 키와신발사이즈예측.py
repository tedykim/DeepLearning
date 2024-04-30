import tensorflow as tf

키 = 170
신발 = 260
# 신발 = a x 키 + b

# 실제값 = 260
# 예측값 = a x 실제값 + b

a = tf.Variable(0.1)  # tf.Variable 은 W 값만드는 문법
b = tf.Variable(0.2)  # 0.1, 0.2 는 일단 임으로 정하면 됨 

def 손실함수():             # 손실함수 = 오류(오차)값
    예측값 = a * 키 + b
 #   실제값 = 260
    return tf.square(260 - 예측값)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)   # a,b를 경사하강법으로 구하기 

for i in range(300):
    opt.minimize(손실함수, var_list=[a,b])    # 경사하강 1번 해줌 == a,b를 한번 수정해줌
    print(a.numpy(),b.numpy())  # print(a,b)  에서 값만 따로 출력하려면 , numpy

# def 신발사이즈(a,b):
#     키 = float(input("키가 몇이죠?"))
#     신발예측 = a*키 + b
#     print(신발예측)

# 신발사이즈()
exit()  

# 이하 내용은 GPT 가 수정했으나, 오류남.
import tensorflow as tf

a = tf.Variable(0.1)  
b = tf.Variable(0.2)  

def loss():             
    prediction = a * height + b
    return tf.square(shoe_size - prediction)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)   

for i in range(300):
    opt.minimize(loss, var_list=[a,b])
    print("Iteration:", i, "a:", a.numpy(), "b:", b.numpy())

# 사용자로부터 키 입력 받기
height = float(input("키를 입력하세요 (cm): "))

# 예측된 신발 크기 출력
predicted_shoe_size = a * height + b
print("예측된 신발 크기:", predicted_shoe_size.numpy())
