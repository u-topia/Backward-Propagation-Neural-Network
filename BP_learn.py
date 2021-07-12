# 三层前馈神经网络的反向传播（BP）算法

import numpy as np 
import random
import math

def menu():
	node_number = []
	# 输入层输入使用input_x表示
	node_number.append(int(input('请输入第一层节点数(输入层)：')))
	node_number.append(int(input('请输入第二层节点数(隐藏层)：')))
	
	node_number.append(int(input('请输入第三层节点数(输出层)：')))
	
	# 输入每一次训练所产生的权值变化量（0.01-0.08之间）
	learn_rate = float(input('请输入学习率：'))
	scope = input('请选择初始化范围(0-1 or -1-1)：')
	activation = input('请输入选择的激活函数（unipolar / bipolar）：')
	return node_number, learn_rate, scope, activation

def Initialize_weights(scope, node_number):
	if scope == '0-1':
		v = np.random.rand(node_number[0], node_number[1])
		w = np.random.rand(node_number[1], node_number[2])
	elif scope == '-1-1':
		v = np.random.uniform(-1, 1, (node_number[0], node_number[1]))
		w = np.random.uniform(-1, 1, (node_number[1], node_number[2]))
	# print(v, w)
	ci = np.zeros([node_number[0], node_number[1]])
	co = np.zeros([node_number[1], node_number[2]])
	return v, w, ci, co

# 单极性sigmoid函数
def sigmoid(t, activation):
	if activation == 'unipolar':
		return 1 / (1 + np.exp(-t))
	elif activation == 'bipolar':
		return math.tanh(t)
		# return (1 - np.exp(-t)) / (1 + np.exp(-t))

def dsigmoid(y, activation):
	if activation == 'unipolar':
		return y * (1 - y)
		# return np.exp(-t) / ((1 + np.exp(-t)) ** 2)
	elif activation == 'bipolar':
		return 1 - y ** 2
		# return 1 - tanh(t / 2) ** 2
		# return (4 * np.exp(-t)) / (1 + np.exp(-t)) ** 2

# 实现前向传播过程
def feed_forward(node_number, activation, v, w, input_x):
	# 激活输入层
	output_x = np.zeros(node_number[0])
	for i in range(node_number[0]):
		output_x[i] = input_x[i]

	# 激活隐藏层
	b_h = np.zeros(node_number[1])
	a_h = np.zeros(node_number[1])
	for i in range(node_number[1]):
		sum = 0
		for j in range(node_number[0]):
			sum += v[j][i] * output_x[j]
		a_h[i] = sum
		b_h[i] = sigmoid(a_h[i], activation)
	# print('b_h', b_h)

	# 激活输出层
	beta_j = np.zeros(node_number[2])
	y_i = np.zeros(node_number[2])
	for i in range(node_number[2]):
		sum = 0
		for j in range(node_number[1]):
			sum += w[j][i] * b_h[j]
		beta_j[i] = sum
		y_i[i] = sigmoid(beta_j[i], activation) 
	# print('y_i', y_i)
	return v, w, b_h, y_i

# 实现反向传播
def back_propagation(node_number, activation, v, w, ci, co, b_h, y_i, learn_rate, d, input_x, config, Ada = False):
	M = 0.1
	# 计算能量函数，即目标函数J,d表示每个样本对应的正确输出
	error = np.zeros(node_number[2])
	# J为目标函数，即所有输出单元的误差
	J = 0
	for i in range(node_number[2]):
		error[i] = d[i] - y_i[i]
		J += error[i] ** 2
	J = 0.5 * J

	# 计算输出层误差,即delta值
	output_delta = np.zeros(node_number[2])
	for j in range(node_number[2]):
		error_1 = d[j] - y_i[j]
		output_delta[j] = dsigmoid(y_i[j], activation) * error_1

	# 计算隐藏层误差，即隐藏层delta值
	hidden_delta = np.zeros(node_number[1])
	for h in range(node_number[1]):
		error_1 = 0.0
		for j in range(node_number[2]):
			error_1 += output_delta[j] * w[h][j]
		hidden_delta[h] = dsigmoid(b_h[h], activation) * error_1

	# 更新输出层权重
	for h in range(node_number[1]):
		for j in range(node_number[2]):
			change = output_delta[j] * b_h[h]
			# print(change)
			if Ada:
				w[h][j], config = adagrad(w[h][j], change, learn_rate, M, co[h][j], config)
			w[h][j] = w[h][j] + learn_rate * change + M * co[h][j]
			co[h][j] = change

	# 更新输入层权重
	for i in range(node_number[0]):
		for h in range(node_number[1]):
			change = hidden_delta[h] * input_x[i]
			if Ada:
				v[i][h], config = adagrad(v[i][h], change, learn_rate, M, ci[i][h], config)
			else:
				v[i][h] = v[i][h] + learn_rate * change + M * ci[i][h]
			ci[i][h] = change

	return v, w, J, config

# 使用自适应学习率进行学习
def adagrad(w, dw, learn_rate,  M, c, config=None):
   
    if config is None: config = {}
    learning_rate=config.get('learning_rate', learn_rate)
    epsilon=config.get('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))
 
    cache=np.zeros_like(w)
    cache +=  dw ** 2
    next_w = w + learning_rate * dw / (np.sqrt(cache) + epsilon) + M * c
 
    config['cache'] = cache
 
    return next_w, config

def test(data, node_number, activation, v, w):
	for d in data:
		v, w, b_h, y_i = feed_forward(node_number, activation, v, w, d[0])
		print(d[0], '->', y_i)

def BP_learn(data):
	node_number, learn_rate, scope, activation = menu()
	v, w, ci, co = Initialize_weights(scope, node_number)
	iterations = 50000
	config = None
	for i in range(iterations):
		error = 0.0
		for d in data:
			inputs = d[0]
			targets = d[1]
			v, w, b_h, y_i = feed_forward(node_number, activation, v, w, inputs)
			v, w, J, config = back_propagation(node_number, activation, v, w, ci, co, b_h, y_i, learn_rate, targets, inputs)
		if i % 100 == 0:
			print('第%d轮误差为%.8f' % (i, J))
		if J <= 0.00001:
			print('在第%d轮达到误差要求' % i)
			break
	test(data, node_number, activation, v, w)

if __name__ == '__main__':
	data = [
	[[1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1], [1, -1, -1]],
	[[-1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1], [-1, 1, -1]],
	[[1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1], [-1, -1, 1]]
	]
	BP_learn(data)
