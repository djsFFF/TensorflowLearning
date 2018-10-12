import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关的常数
INPUT_NODE = 784
OUTPUT_NODE = 10

# 一个有500个结点的隐藏层
LAYER1_NODE = 500
# 一个训练batch中的训练数据个数
BATCH_SIZE = 100
# 基础的学习率
LEARNING_RATE_BASE = 0.8
# 学习率的衰减
LEARNING_RATE_DECAY = 0.99
# 描述模型复杂度的正则化项在损失函数中的系数
REGULARIZATION_RATE = 0.001
# 训练轮数
TRAINING_STEP = 30000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99

'''
定义了一个使用ReLU激活函数的三层全连接神经网络
'''
def inference(input_tensor, reuse = False):
	with tf.variable_scope('layer1', reuse = reuse):
		weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE], initializer = tf.truncated_normal_initializer(stddev = 0.1))
		biases = tf.get_variable("biases", [LAYER1_NODE], initializer = tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

	with tf.variable_scope('layer2', reuse = reuse):
		weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE], initializer = tf.truncated_normal_initializer(stddev = 0.1))
		biases = tf.get_variable("biases", [OUTPUT_NODE], initializer = tf.constant_initializer(0.0))
		layer2 = tf.matmul(layer1, weights) + biases
	return layer2

'''
训练模型的过程
'''
def train(mnist):
	x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
	y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y-input')

	# 生成隐藏层的参数
	weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev = 0.1))
	biases1 = tf.Variable(tf.constant(0.1, shape = [LAYER1_NODE]))
	# 生成输出层的参数
	weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev = 0.1))
	biases2 = tf.Variable(tf.constant(0.1, shape = [OUTPUT_NODE]))

	y = inference(x)
	# 存储训练轮数的变量
	global_step = tf.Variable(0, trainable = False)
	# 初始化滑动平均类
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variable_averages_op = variable_averages.apply(tf.trainable_variables())
	average_y = inference(x, True)

	# 计算交叉熵
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)

	# 计算L2正则化损失函数
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	# 计算模型正则化损失
	regularization = regularizer(weights1) + regularizer(weights2)
	# 总损失等于交叉熵损失和正则化损失的和
	loss = cross_entropy_mean + regularization

	# 设置指数衰减的学习率
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, 
		global_step, 
		mnist.train.num_examples / BATCH_SIZE,
		LEARNING_RATE_DECAY)

	# 优化损失函数
	train_step = tf.train.GradientDescentOptimizer(learning_rate) \
				   .minimize(loss, global_step = global_step)

	with tf.control_dependencies([train_step, variable_averages_op]):
		train_op = tf.no_op(name = 'train')

	# 检验使用了滑动平均模型的神经网络前向传播结果是否正确
	correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# 初始化会话并开始训练过程
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		# 准备验证数据
		validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
		# 准备测试数据
		test_feed = {x: mnist.test.images, y_:mnist.test.labels}

		# 迭代地训练神经网络
		for i in range(TRAINING_STEP):
			if i % 1000 == 0:
				validate_acc = sess.run(accuracy, feed_dict = validate_feed)
				test_acc = sess.run(accuracy, feed_dict = test_feed)
				print("After %d training step(s), validation accuracy using average "
					  "model is %g, test accuracy using average model is %g" % (i, validate_acc, test_acc))
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			sess.run(train_op, feed_dict = {x: xs, y_: ys})

		# 训练结束之后，在测试数据上检测神经网络模型的最终正确率
		# test_acc = sess.run(accuracy, feed_dict = test_feed)
		# print("After %d training step(s), test accuracy using average "
		# 	  "model is %g" % (TRAINING_STEP, test_acc))

def main(argv = None):
	mnist = input_data.read_data_sets("E:/tensorflow/MNIST_data/", one_hot = True)
	train(mnist)

if __name__ == '__main__':
	tf.app.run()