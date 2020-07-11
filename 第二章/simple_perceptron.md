# 使用Python（不使用Numpy）实现一个简单的感知器
思路：只设一个权重w，一个bias参数，network_input为wx+bias（bias也可视为权重w0，对应一个输入为1的常数。）φ(z)作为激活函数，根据网络输入的值来输出1或0。
```py
class Perceptron(object):
	def __init__(self, eta=0.01, iterations=10):
		self.lr = eta#learning rate学习率，用于调整训练时的步长
		self.iterations = iterations#迭代次数
		self.w = 0.0
		self.bias = 0.0


	def fit(self, X, Y):
		self.errors = []

		for _ in range(self.iterations):
			error = 0
			for i in range(len(X)):
				x = X[i]
				y = Y[i]
				update = self.lr * (y - self.predict(x))
				self.w += update * x
				self.bias += update
				error += int(update != 0.0)
			self.errors.append(error)


	def net_input(self, x):
		return self.w * x + self.bias


	def predict(self, x):
		return 1.0 if self.net_input(x) > 0.0 else 0.0



x = [1, 2, 3, 10, 20, -2, -10, -100, -5, -20]
y = [1.0, 1.0, 1.0, 1.0, 1.0,  0.0, 0.0, 0.0, 0.0, 0.0]

model = Perceptron(0.01, 10)
model.fit(x, y)

test_x = [30, 40, -20, -60]
for i in range(len(test_x)):
	print('input {} => predict: {}'.format(test_x[i], model.predict(test_x[i])))

print(model.w)
print(model.bias)
```

运行结果：
```
input 30 => predict: 1.0
input 40 => predict: 1.0
input -20 => predict: 0.0
input -60 => predict: 0.0
0.01
0.01
```