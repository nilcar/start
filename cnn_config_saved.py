
import sys

import tensorflow


	
def cnn_model_dnn5_fn(features, labels, mode):
	
	print('##### CNN-DNN5')
	
	"""Model function for CNN."""
	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]

	input_layer = tensorflow.reshape(features["x"], [-1, 20, 20, 1])

	# Convolutional Layer #1
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 20, 20, 1]
	# Output Tensor Shape: [batch_size, 20, 20, 32]
	
	conv1 = tensorflow.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tensorflow.nn.relu)
	
	#conv1bn = tensorflow.layers.batch_normalization(conv1, training=mode == tensorflow.estimator.ModeKeys.TRAIN)
	
	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 20, 20, 32]
	# Output Tensor Shape: [batch_size, 10, 10, 32]
	#pool1 = tensorflow.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 20, 20, 32]
	# Output Tensor Shape: [batch_size, 20, 20, 64]
	#conv2 = tensorflow.layers.conv2d(inputs=conv1bn, filters=64, kernel_size=[5, 5], padding="same", activation=tensorflow.nn.relu)
	
	conv2dil = tensorflow.layers.conv2d(inputs=conv1, filters=64, dilation_rate=[2, 2], kernel_size=[5, 5], padding="same", activation=tensorflow.nn.relu)
	
	#print('Shape after conv2: ', conv2dil.shape)
	
	#sys.exit()
	
	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 10, 10, 64]
	# Output Tensor Shape: [batch_size, 5, 5, 64]
	#pool2 = tensorflow.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, 20, 20, 64]
	# Output Tensor Shape: [batch_size, 20 * 20 * 64]
	pool2_flat = tensorflow.reshape(conv2dil, [-1, 20 * 20 * 64])

	# Dense Layers
	# Input Tensor Shape: [batch_size, 20 * 20 * 64]
	dense = tensorflow.layers.dense(inputs=pool2_flat, units=800, activation=tensorflow.nn.relu)
	
	dropout = tensorflow.layers.dropout(inputs=dense, rate=0.2, training=mode == tensorflow.estimator.ModeKeys.TRAIN)
	
	dense1 = tensorflow.layers.dense(inputs=dropout, units=800, activation=tensorflow.nn.relu)
	
	dropout2 = tensorflow.layers.dropout(inputs=dense1, rate=0.2, training=mode == tensorflow.estimator.ModeKeys.TRAIN)
	
	dense2 = tensorflow.layers.dense(inputs=dropout2, units=400, activation=tensorflow.nn.relu)
	
	dropout3 = tensorflow.layers.dropout(inputs=dense2, rate=0.2, training=mode == tensorflow.estimator.ModeKeys.TRAIN)
	
	dense3 = tensorflow.layers.dense(inputs=dropout3, units=400, activation=tensorflow.nn.relu)
	
	dropout4 = tensorflow.layers.dropout(inputs=dense3, rate=0.2, training=mode == tensorflow.estimator.ModeKeys.TRAIN)
	
	dense4 = tensorflow.layers.dense(inputs=dropout4, units=20, activation=tensorflow.nn.relu)
	# Output Tensor Shape: [batch_size, 20]
	
	# Logits layer
	# Input Tensor Shape: [batch_size, 20]
	# Output Tensor Shape: [batch_size, 2]
	logits = tensorflow.layers.dense(inputs=dense4, units=2, activation=tensorflow.nn.softmax)

	# Generate predictions (for PREDICT and EVAL mode)
	# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
	# `logging_hook`.
	predictions = {"class_ids": tensorflow.argmax(input=logits, axis=1),"probabilities": tensorflow.nn.softmax(logits, name="softmax_tensor")}
	if mode == tensorflow.estimator.ModeKeys.PREDICT:
		return tensorflow.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tensorflow.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tensorflow.estimator.ModeKeys.TRAIN:
		optimizer = tensorflow.train.AdagradOptimizer(learning_rate=0.01)
		
		#update_ops = tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS) # For BN
		train_op = optimizer.minimize(loss=loss, global_step=tensorflow.train.get_global_step())
		#train_op = tensorflow.group([train_op, update_ops]) # For BN
		return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tensorflow.metrics.accuracy(labels=labels, predictions=predictions["class_ids"]),
		"precision": tensorflow.metrics.precision( labels=labels, predictions=predictions["class_ids"] ),
		"recall":    tensorflow.metrics.recall(    labels=labels, predictions=predictions["class_ids"] ),
		"auc":    tensorflow.metrics.auc(    labels=labels, predictions=predictions["class_ids"] )}
	return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
	
	
def cnn_model_dnn5CL3_fn(features, labels, mode):
	
	print('##### CNN-DNN5')
	
	"""Model function for CNN."""
	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]

	input_layer = tensorflow.reshape(features["x"], [-1, 20, 20, 1])

	# Convolutional Layer #1
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 20, 20, 1]
	# Output Tensor Shape: [batch_size, 20, 20, 32]
	
	conv1 = tensorflow.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tensorflow.nn.relu)
	
	#conv1bn = tensorflow.layers.batch_normalization(conv1, training=mode == tensorflow.estimator.ModeKeys.TRAIN)
	
	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 20, 20, 32]
	# Output Tensor Shape: [batch_size, 10, 10, 32]
	#pool1 = tensorflow.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 10, 10, 32]
	# Output Tensor Shape: [batch_size, 10, 10, 64]
	#conv2 = tensorflow.layers.conv2d(inputs=conv1bn, filters=64, kernel_size=[5, 5], padding="same", activation=tensorflow.nn.relu)
	
	conv2 = tensorflow.layers.conv2d(inputs=conv1, filters=64, kernel_size=[5, 5], padding="same", activation=tensorflow.nn.relu)
	
	conv2dil = tensorflow.layers.conv2d(inputs=conv2, filters=128, dilation_rate=[2, 2], kernel_size=[5, 5], padding="same", activation=tensorflow.nn.relu)
	
	
	#print('Shape after conv2: ', conv2dil.shape)
	
	#sys.exit()
	
	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 10, 10, 64]
	# Output Tensor Shape: [batch_size, 5, 5, 64]
	#pool2 = tensorflow.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, 20, 20, 64]
	# Output Tensor Shape: [batch_size, 20 * 20 * 64]
	pool2_flat = tensorflow.reshape(conv2dil, [-1, 20 * 20 * 128])

	# Dense Layers
	# Input Tensor Shape: [batch_size, 20 * 20 * 64]
	dense = tensorflow.layers.dense(inputs=pool2_flat, units=800, activation=tensorflow.nn.relu)
	
	# Add dropout operation; 0.8 probability that element will be kept
	dropout = tensorflow.layers.dropout(inputs=dense, rate=0.2, training=mode == tensorflow.estimator.ModeKeys.TRAIN)
	
	dense1 = tensorflow.layers.dense(inputs=dropout, units=800, activation=tensorflow.nn.relu)
	
	dropout2 = tensorflow.layers.dropout(inputs=dense1, rate=0.2, training=mode == tensorflow.estimator.ModeKeys.TRAIN)
	
	dense2 = tensorflow.layers.dense(inputs=dropout2, units=400, activation=tensorflow.nn.relu)
	
	dropout3 = tensorflow.layers.dropout(inputs=dense2, rate=0.2, training=mode == tensorflow.estimator.ModeKeys.TRAIN)
	
	dense3 = tensorflow.layers.dense(inputs=dropout3, units=400, activation=tensorflow.nn.relu)
	
	dropout4 = tensorflow.layers.dropout(inputs=dense3, rate=0.2, training=mode == tensorflow.estimator.ModeKeys.TRAIN)
	
	dense4 = tensorflow.layers.dense(inputs=dropout4, units=20, activation=tensorflow.nn.relu)
	# Output Tensor Shape: [batch_size, 20]

	# Logits layer
	# Input Tensor Shape: [batch_size, 20]
	# Output Tensor Shape: [batch_size, 2]
	logits = tensorflow.layers.dense(inputs=dense4, units=2) # , activation=tensorflow.nn.softmax

	# Generate predictions (for PREDICT and EVAL mode)
	# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
	# `logging_hook`.
	predictions = {"class_ids": tensorflow.argmax(input=logits, axis=1),"probabilities": tensorflow.nn.softmax(logits, name="softmax_tensor")}
	if mode == tensorflow.estimator.ModeKeys.PREDICT:
		return tensorflow.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tensorflow.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tensorflow.estimator.ModeKeys.TRAIN:
		#optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.01)
		#optimizer = tensorflow.train.ProximalAdagradOptimizer(learning_rate=0.01, l1_regularization_strength=0.01)
		optimizer = tensorflow.train.AdagradOptimizer(learning_rate=0.01)
		#optimizer = tensorflow.train.AdamOptimizer(learning_rate=0.01)
		
		#update_ops = tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS) # For BN
		train_op = optimizer.minimize(loss=loss, global_step=tensorflow.train.get_global_step())
		#train_op = tensorflow.group([train_op, update_ops]) # For BN
		return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tensorflow.metrics.accuracy(labels=labels, predictions=predictions["class_ids"]),
		"precision": tensorflow.metrics.precision( labels=labels, predictions=predictions["class_ids"] ),
		"recall":    tensorflow.metrics.recall(    labels=labels, predictions=predictions["class_ids"] ),
		"auc":    tensorflow.metrics.auc(    labels=labels, predictions=predictions["class_ids"] )}
	return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
	
	