########################################################
###### Example: EfficientNetB0 for Stanford Dogs. ######
########################################################


# ERROR : his TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags. <- 해결하는 코드
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print(X_train_full.shape, X_train_full.dtype)
print(y_train_full.shape, y_train_full.dtype)

print(X_test.shape, X_test.dtype)
print(y_test.shape, y_test.dtype)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(class_names[y_train[0]], y_train[0])

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))

model.add(keras.layers.Dense(100, activation="relu"))

model.add(keras.layers.Dense(10, activation="softmax"))

# print(model.layers)

# bias = 편향, weights = 가중치
# 시퀸셜 레이어는 weight값이 존재하지 않음
hidden1 = model.layers[1]
# weights, biases = hidden1.get_weights()
# print(weights, biases)
# print(weights.shape, biases.shape)
# a = 0
# for i in weights:
#     for j in i:
#         a+=j
#     print(a)
#     a = 0
# print(a)

hidden1 = model.layers[2]
# weights, biases = hidden1.get_weights()
# print(weights, biases)
# print(weights.shape, biases.shape)
# a = 0
# for i in weights:
#     for j in i:
#         a+=j
# print(a)

hidden1 = model.layers[3]
# weights, biases = hidden1.get_weights()
# # print(weights, biases)
# print(weights.shape, biases.shape)
# a = 0
# for i in weights:
#     for j in i:
#         a+=j
# print(a)

# sgd = 기본 확률적 경사 하강법을 사용해서 모델 훈련
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

epochs_num = 5
history = model.fit(X_train, y_train, epochs=epochs_num, validation_data=(X_valid, y_valid))


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.gca().set_xlim(0, epochs_num)
plt.show()

# 테스트 모델 평가하여 일반화 오차를 추정하는 함수
print("model.evaluate : ", model.evaluate(X_test, y_test))

# 모델 사용해 예측 만들기
X_new = X_test[:3]
y_proba = model.predict(X_new)
print("y_proba.round(2) : ",y_proba.round(2))

# 가장 높은 확률을 가진 클래스에만 관심