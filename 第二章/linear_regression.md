# 神经元的线性回归实现
```py
class LinearRegression(object):
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

        for i in range(n):
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

model = LinearRegression(0.01, 500)

X = [(k+100)/200 for k in x] #数据归一化

model.fit(X, y)

test_x = [90, 80,81, 82, 75, 40, 32, 15, 5, 1, -1, -15, -20, -22, -33, -45, -60, -90]
for i in range(len(test_x)):
    print('input {} => predict: {}'.format(test_x[i], model.predict(test_x[i])))

# print(model.w)
# print(model.bias)
```

损失函数即如何计算Error值。  
$$MSE = \frac{\sum_{i=1}^n(y_i - y_i')^2}{N} = \frac{\sum_{i=1}^n(y_i - (wx_i + b))^2}{N}$$
假设f为MSE，分别对w和b求导：  
$$\Delta w = \frac{\partial f}{\partial w} = \frac{\partial \frac{\sum_{i=1}^n(y_i - (wx_i + b))^2}{N}}{\partial w} = \frac{\sum_{i=1}^n-2x_i(y_i - (wx_i + b))}{N}$$
$$\Delta b = \frac{\partial f}{\partial b} = \frac{\partial \frac{\sum_{i=1}^n(y_i - (wx_i + b))^2}{N}}{\partial w} = \frac{\sum_{i=1}^n-2(y_i - (wx_i + b))}{N}$$
以上即梯度$\Delta w$和$\Delta b$，然后只需更新w(w - $\Delta w$)、b(b - \Delta b)即可。但在代码中没有直接从w和bias中减去梯度值，而是将梯度值乘以learning rate，以调整训练步长。

数据归一化公式：
$$x_i = \frac{x_i - x_{min}}{x_{max} - x_{min}}$$

运行结果：
```
iter=0    weight=0.01    bias=0.0117    cost=0.57
iter=10    weight=0.06    bias=0.1130    cost=0.43
iter=20    weight=0.11    bias=0.1909    cost=0.35
iter=30    weight=0.14    bias=0.2508    cost=0.3
iter=40    weight=0.17    bias=0.2968    cost=0.28
iter=50    weight=0.19    bias=0.3321    cost=0.26
iter=60    weight=0.21    bias=0.3591    cost=0.25
iter=70    weight=0.22    bias=0.3798    cost=0.24
iter=80    weight=0.23    bias=0.3955    cost=0.24
iter=90    weight=0.24    bias=0.4074    cost=0.24
iter=100    weight=0.25    bias=0.4164    cost=0.24
iter=110    weight=0.25    bias=0.4232    cost=0.24
iter=120    weight=0.26    bias=0.4282    cost=0.23
iter=130    weight=0.26    bias=0.4319    cost=0.23
iter=140    weight=0.26    bias=0.4346    cost=0.23
iter=150    weight=0.27    bias=0.4364    cost=0.23
iter=160    weight=0.27    bias=0.4377    cost=0.23
iter=170    weight=0.27    bias=0.4385    cost=0.23
iter=180    weight=0.28    bias=0.4389    cost=0.23
iter=190    weight=0.28    bias=0.4391    cost=0.23
iter=200    weight=0.28    bias=0.4390    cost=0.23
iter=210    weight=0.28    bias=0.4388    cost=0.23
iter=220    weight=0.28    bias=0.4384    cost=0.23
iter=230    weight=0.28    bias=0.4380    cost=0.23
iter=240    weight=0.29    bias=0.4374    cost=0.23
iter=250    weight=0.29    bias=0.4369    cost=0.23
iter=260    weight=0.29    bias=0.4363    cost=0.23
iter=270    weight=0.29    bias=0.4356    cost=0.23
iter=280    weight=0.29    bias=0.4350    cost=0.23
iter=290    weight=0.29    bias=0.4343    cost=0.23
iter=300    weight=0.29    bias=0.4336    cost=0.23
iter=310    weight=0.30    bias=0.4329    cost=0.23
iter=320    weight=0.30    bias=0.4322    cost=0.23
iter=330    weight=0.30    bias=0.4315    cost=0.23
iter=340    weight=0.30    bias=0.4308    cost=0.23
iter=350    weight=0.30    bias=0.4301    cost=0.23
iter=360    weight=0.30    bias=0.4294    cost=0.23
iter=370    weight=0.30    bias=0.4287    cost=0.23
iter=380    weight=0.31    bias=0.4280    cost=0.23
iter=390    weight=0.31    bias=0.4273    cost=0.23
iter=400    weight=0.31    bias=0.4266    cost=0.23
iter=410    weight=0.31    bias=0.4260    cost=0.23
iter=420    weight=0.31    bias=0.4253    cost=0.23
iter=430    weight=0.31    bias=0.4246    cost=0.23
iter=440    weight=0.31    bias=0.4239    cost=0.23
iter=450    weight=0.31    bias=0.4233    cost=0.23
iter=460    weight=0.32    bias=0.4226    cost=0.23
iter=470    weight=0.32    bias=0.4220    cost=0.23
iter=480    weight=0.32    bias=0.4213    cost=0.23
iter=490    weight=0.32    bias=0.4207    cost=0.23
input 90 => predict: 0.7238852731029148
input 80 => predict: 0.7078964168036423
input 81 => predict: 0.7094953024335697
input 82 => predict: 0.7110941880634969
input 75 => predict: 0.6999019886540062
input 40 => predict: 0.6439409916065527
input 32 => predict: 0.6311499065671348
input 15 => predict: 0.6039688508583716
input 5 => predict: 0.5879799945590992
input 1 => predict: 0.5815844520393902
input -1 => predict: 0.5783866807795357
input -15 => predict: 0.5560022819605543
input -20 => predict: 0.5480078538109181
input -22 => predict: 0.5448100825510637
input -33 => predict: 0.5272223406218639
input -45 => predict: 0.5080357130627371
input -60 => predict: 0.4840524286138284
input -90 => predict: 0.4360858597160111
```
对于所有测试数据，可以看到阈值基本上在0.58左右，正数越大越趋近于1，负数越小越趋近于0。