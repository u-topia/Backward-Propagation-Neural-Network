import matplotlib.pyplot as plt
import numpy as np

def draw_graph():
	y = []
	x = []
	for i in range(4, 11, 1):
		filename = 'test_num_of_hidden' + str(i) + '.txt'
		y_1 = []
		x_1 = []
		with open(filename, 'r') as f:
			for line in f:
				data = line.split('\t')
				# print(data)
				if int(data[0]) == 1:
					continue
				if int(data[0]) > 2001:
					break
				y_1.append(float(data[1].replace('\n', '')))
				x_1.append(int(data[0]))
		y.append(y_1)
		x.append(x_1)
	# 绘制图像
	line_label = [4, 5, 6, 7, 8 ,9, 10]
	for i in range(len(x)):
		plt.plot(x[i], y[i], label = line_label[i])

	plt.title('Square error variation graph')
	plt.xlabel('Number of rounds')
	plt.ylabel('Loss')

	plt.legend()

	plt.show()

def draw_ada():
	# print(1)
	y1 = [5.23426797, 3.47548730, 2.26847378, 1.92910769, 0.98434275,0.00000385]
	x1 = [1, 2, 3, 4, 5, 6]
	y2 = [5.23426797, 3.17909556, 0.00613036, 0.00000097]
	x2 = [1, 2, 3, 4]

	line_label = ['non_ada', 'ada']
	plt.plot(x1, y1, label = line_label[0])
	plt.plot(x2, y2, label = line_label[1])

	plt.title('Constant learning rate and adaptive learning rate')
	plt.xlabel('Number of rounds')
	plt.ylabel('Loss')

	plt.legend()

	plt.show()


if __name__ == '__main__':
	# draw_graph()
	draw_ada()