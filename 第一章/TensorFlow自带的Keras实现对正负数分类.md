# TensorFlow自带的Keras实现对正负数分类
```py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()#按顺序层叠的网络
model.add(Dense(units=8, activation='relu', input_dim=1))#1个输入 8个输出（可视为神经元的个数）
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='sgd')#损失函数计算为平均方差，梯度优化方式为随机梯度下降（Stochastic Gradient Descent, SGD）

x = [1, 2, 3, 10, 20, -2, -10, -100, -5, -20]
y = [1.0, 1.0, 1.0, 1.0, 1.0,  0.0, 0.0, 0.0, 0.0, 0.0]#分类结果
model.fit(x, y, epochs=10, batch_size=4)#训练10次，随机挑选4组数据

test_x = [30, 40, -20, -60]
test_y = model.predict(test_x)

for i in range(len(test_x)):
	print('input {} => predict: {}'.format(test_x[i], test_y[i]))
```
运行结果
```
Epoch 1/10

 4/10 [===========>..................] - ETA: 0s - loss: 0.1407
10/10 [==============================] - 1s 54ms/sample - loss: 0.1357
Epoch 2/10

 4/10 [===========>..................] - ETA: 0s - loss: 0.1161
10/10 [==============================] - 0s 300us/sample - loss: 0.1177
Epoch 3/10

 4/10 [===========>..................] - ETA: 0s - loss: 0.0820
10/10 [==============================] - 0s 200us/sample - loss: 0.1092
Epoch 4/10

 4/10 [===========>..................] - ETA: 0s - loss: 0.1730
10/10 [==============================] - 0s 200us/sample - loss: 0.1035
Epoch 5/10

 4/10 [===========>..................] - ETA: 0s - loss: 0.1030
10/10 [==============================] - 0s 200us/sample - loss: 0.0987
Epoch 6/10

 4/10 [===========>..................] - ETA: 0s - loss: 0.0853
10/10 [==============================] - 0s 200us/sample - loss: 0.0947
Epoch 7/10

 4/10 [===========>..................] - ETA: 0s - loss: 0.1189
10/10 [==============================] - 0s 200us/sample - loss: 0.0906
Epoch 8/10

 4/10 [===========>..................] - ETA: 0s - loss: 0.0938
10/10 [==============================] - 0s 200us/sample - loss: 0.0880
Epoch 9/10

 4/10 [===========>..................] - ETA: 0s - loss: 0.1077
10/10 [==============================] - 0s 200us/sample - loss: 0.0857
Epoch 10/10

 4/10 [===========>..................] - ETA: 0s - loss: 0.1267
10/10 [==============================] - 0s 200us/sample - loss: 0.0835
input 30 => predict: [0.99732506]
input 40 => predict: [0.9996259]
input -20 => predict: [0.04541726]
input -60 => predict: [0.00011078]
```
运行结果可得，所有正数的预测值都接近1，负数预测值都接近0。