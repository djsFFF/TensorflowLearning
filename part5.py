# 通过集合计算一个5层神经网络带L2正则化的损失函数的计算方法

import tensorflow as tf
# 获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为'losses'的集合中
def get_weight(shape, lambda):
	# 生成一个变量
	var = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
	tf.add_to_collection(
		'losses', tf.contrib.layers.l2_regularizer(lambda)(var))
	return var

x = tf.placeholder(tf.float32, shape = (None, 2))
y_ = tf.placeholder(tf.float32, shape = (None, 1))
batch_size = 8
layer_dimension = [2, 10, 10, 10, 1]
n_layers = len(layer_dimension)

cur_layer = x
in_dimension = layer_dimension[0]

for i in range(1, n_layers):
	out_dimension = layer_dimension[i]
	weight = get_weight([in_dimension, out_dimension], 0.001)
	bias = tf.Variable(tf.constant(0.1, shape = [out_dimension]))
	cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
	in_dimension = layer_dimension[i]


mse_loss = tf.reduce_mean(tf.sequare(y_ - cur_layer))

tf.add_to_collection('losses', mse_loss)

loss = tf.add_n(tf.get_collection('losses'))