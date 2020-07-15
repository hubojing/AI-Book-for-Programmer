# 随机梯度下降
update_weights方法如果每次都载入所有数据进行训练，在实际生产环境中不现实，通常会采用mini batch方法，即每次更新权重时既不使用所有数据，也不随机挑选一条数据，而是随机挑选一个子集训练。mini batch的大小设置和模型、数据特点等都有关系。  
若batch_size为1，就是随机梯度下降。
```py
import random

class LinearRegressionMiniBatch(object):
    def __init__(self, eta=0.01, iterations=10):
        self.lr = eta
        self.iterations = iterations
        self.w = 0.0
        self.bias = 0.0

    def cost_function(self, X, Y, weight, bias):
        n = len(X)
        total_error = 0.0
        for i in range(n):
            total_error += (Y[i] - (weight*X[i] + bias))**2
        return total_error / n


    def update_weights(self, X, Y, weight, bias, learning_rate):
        dw = 0
        db = 0
        n = len(X)

        indexes = list(range(0, n))
        random.shuffle(indexes)
        batch_size = 4

        for k in range(batch_size):
            i = indexes[k]
            dw += -2 * X[i] * (Y[i] - (weight*X[i] + bias))
            db += -2 * (Y[i] - (weight*X[i] + bias))

        weight -= (dw / n) * learning_rate
        bias -= (db / n) * learning_rate

        return weight, bias


    def fit(self, X, Y):
        cost_history = []

        for i in range(self.iterations):
            self.w, self.bias = self.update_weights(X, Y, self.w, self.bias, self.lr)

            #Calculate cost for auditing purposes
            cost = self.cost_function(X, Y, self.w, self.bias)
            cost_history.append(cost)

            # Log Progress
            if i % 10 == 0:
                print("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, self.w, self.bias, cost))

        return self.w, self.bias, cost_history

    def predict(self, x):
        x = (x+100)/200
        return self.w * x + self.bias


x = [1, 2, 3, 10, 20, 50, 100, -2, -10, -100, -5, -20]
y = [1.0, 1.0, 1.0, 1.0, 1.0,  1,0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

model = LinearRegressionMiniBatch(0.01, 500)

X = [(k+100)/200 for k in x]

model.fit(X, y)

test_x = [90, 80,81, 82, 75, 40, 32, 15, 5, 1, -1, -15, -20, -22, -33, -45, -60, -90]
for i in range(len(test_x)):
    print('input {} => predict: {}'.format(test_x[i], model.predict(test_x[i])))

# print(model.w)
# print(model.bias)
```
运行结果：
```
iter=0    weight=0.00    bias=0.0033    cost=0.58
iter=10    weight=0.02    bias=0.0399    cost=0.53
iter=20    weight=0.04    bias=0.0720    cost=0.48
iter=30    weight=0.06    bias=0.1043    cost=0.44
iter=40    weight=0.07    bias=0.1277    cost=0.41
iter=50    weight=0.09    bias=0.1522    cost=0.39
iter=60    weight=0.10    bias=0.1672    cost=0.37
iter=70    weight=0.11    bias=0.1883    cost=0.35
iter=80    weight=0.12    bias=0.2091    cost=0.34
iter=90    weight=0.13    bias=0.2271    cost=0.32
iter=100    weight=0.14    bias=0.2440    cost=0.31
iter=110    weight=0.15    bias=0.2672    cost=0.29
iter=120    weight=0.16    bias=0.2806    cost=0.29
iter=130    weight=0.16    bias=0.2942    cost=0.28
iter=140    weight=0.17    bias=0.2993    cost=0.28
iter=150    weight=0.17    bias=0.3070    cost=0.27
iter=160    weight=0.18    bias=0.3246    cost=0.26
iter=170    weight=0.19    bias=0.3320    cost=0.26
iter=180    weight=0.20    bias=0.3427    cost=0.25
iter=190    weight=0.20    bias=0.3530    cost=0.25
iter=200    weight=0.21    bias=0.3602    cost=0.25
iter=210    weight=0.21    bias=0.3701    cost=0.25
iter=220    weight=0.22    bias=0.3726    cost=0.25
iter=230    weight=0.22    bias=0.3732    cost=0.24
iter=240    weight=0.22    bias=0.3854    cost=0.24
iter=250    weight=0.23    bias=0.3931    cost=0.24
iter=260    weight=0.23    bias=0.4014    cost=0.24
iter=270    weight=0.24    bias=0.4080    cost=0.24
iter=280    weight=0.25    bias=0.4219    cost=0.24
iter=290    weight=0.25    bias=0.4203    cost=0.24
iter=300    weight=0.25    bias=0.4202    cost=0.24
iter=310    weight=0.25    bias=0.4248    cost=0.24
iter=320    weight=0.25    bias=0.4274    cost=0.24
iter=330    weight=0.26    bias=0.4306    cost=0.23
iter=340    weight=0.26    bias=0.4365    cost=0.23
iter=350    weight=0.26    bias=0.4400    cost=0.23
iter=360    weight=0.26    bias=0.4407    cost=0.23
iter=370    weight=0.27    bias=0.4472    cost=0.23
iter=380    weight=0.27    bias=0.4513    cost=0.23
iter=390    weight=0.27    bias=0.4540    cost=0.23
iter=400    weight=0.28    bias=0.4606    cost=0.23
iter=410    weight=0.27    bias=0.4586    cost=0.23
iter=420    weight=0.27    bias=0.4564    cost=0.23
iter=430    weight=0.27    bias=0.4508    cost=0.23
iter=440    weight=0.27    bias=0.4488    cost=0.23
iter=450    weight=0.27    bias=0.4477    cost=0.23
iter=460    weight=0.27    bias=0.4515    cost=0.23
iter=470    weight=0.27    bias=0.4453    cost=0.23
iter=480    weight=0.27    bias=0.4466    cost=0.23
iter=490    weight=0.27    bias=0.4451    cost=0.23
input 90 => predict: 0.7011140587195266
input 80 => predict: 0.6875398466801047
input 81 => predict: 0.6888972678840469
input 82 => predict: 0.690254689087989
input 75 => predict: 0.6807527406603937
input 40 => predict: 0.6332429985224168
input 32 => predict: 0.6223836288908793
input 15 => predict: 0.5993074684238618
input 5 => predict: 0.58573325638444
input 1 => predict: 0.5803035715686711
input -1 => predict: 0.5775887291607867
input -15 => predict: 0.558584832305596
input -20 => predict: 0.551797726285885
input -22 => predict: 0.5490828838780005
input -33 => predict: 0.5341512506346364
input -45 => predict: 0.51786219618733
input -60 => predict: 0.4975008781281971
input -90 => predict: 0.45677824200993117
```
可以看出预测效果差别不大。