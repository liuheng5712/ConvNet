import scipy.io as sio
import numpy as np
import os
import tensorflow as tf


def ConvNet(X, Y, mode='train', layer_num = 2, filter_num=[32, 64], kernel = (5,5), poolsize=2, stride = 2, Dneuron_num=1024, dropout_rate=0.2, scope_name = 'vars'):
    """
    This function creates the ConvNet graph for given specifications, 
    X:             input data that is 4-dim np array that has shape [batch_size, height, width, channels];
    Y:             the 2-dim label vector [batch_size, 3]; 
    layer_num:     number of convolutional layers; 
    filter_num:    number of filters at each convolutional layer, the length equals layer_num
    kernel_size:   the size of filter kernel, we use same size for each layer
    Dneuron_num:   the neuron number of the final dense layer 
    dropout_rate:  dropout rate
    """    
    with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE):
        # conv layers
        conv1 = tf.layers.conv2d(inputs = X, filters = filter_num[0], kernel_size=kernel, padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=poolsize, strides=stride)
        conv2 = tf.layers.conv2d(inputs = pool1, filters = filter_num[1], kernel_size=kernel, padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size=poolsize, strides=stride)
        # flatten layer
        flat_layer = tf.reshape(pool2, [-1, np.prod(pool2.get_shape().as_list()[1:])])
        dense_layer = tf.layers.dense(inputs=flat_layer, units=Dneuron_num, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        # The final output layer: regression layer
        logits = tf.layers.dense(inputs = dense_layer, units = 3, activation = tf.keras.activations.linear)
    
    # if is the predict mode, we stop here
    if mode == 'predict':   
        return logits  
    # if train mode, we continue
    train_loss = tf.losses.mean_squared_error(Y, logits)
    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(learning_rate = 3e-4)
        train_op = optimizer.minimize(loss = train_loss, global_step = tf.train.get_global_step())
        return train_loss, logits, train_op

