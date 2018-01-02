# coding=utf-8

import numpy as np

class LinearRegression:
	
	def __init__(self):
		self.eta = 0.01
		self.b = 0
	
	def fit(self, train_x, train_y, epochs):
		self.w = np.random.normal(size = train_x.shape[1])
		for i in range(epochs):
			w_gradient, b_gradient, mse = self.gradient(train_x, train_y)
			self.w -= self.eta * w_gradient
			self.b -= self.eta * b_gradient
			print 'Epochs {0}/{1}, mse = {2}.'.format(i + 1, epochs, mse) 
	
	def gradient(self, train_x, train_y):
		"""
		计算mse的值,以及w和b的偏微分值
		返回的是所有样本上的平均值
		"""
		size = len(train_x)
		total_se = 0
		w_gradients = np.zeros(train_x.shape[1])
		b_gradients = 0
		for i in range(size):
			x = train_x[i]
			y = train_y[i]
			y_hat = self.predict(x)
			total_se += self.SE(y, y_hat)
			w_gradients += (y_hat - y) * x
			b_gradients += (y_hat - y)
		return 	w_gradients / float(size), b_gradients / float(size), total_se / float(size)
	
	def SE(self, y, y_hat):
		return (y_hat - y) ** 2
	
	def predict(self, x):
		return np.dot(self.w, x) + self.b
	

if __name__ == "__main__":
	raw_data = np.loadtxt('lr.train')
	train_x = raw_data[:,:3]
	train_y = raw_data[:, 3:]
	train_y = np.reshape(train_y, (len(train_y)))
	linear = LinearRegression()
	linear.fit(train_x, train_y, 50)
	y_hat = linear.predict(np.array([4, 5, 6]))
	print linear.w, linear.b
	print y_hat
		
