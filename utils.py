import scipy.io as sio
import numpy as np
import os
import tensorflow as tf
from models import ConvNet

from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa

def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]

def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)
    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)
    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def Trainer(sess, training_size = 5000, batch_size = 32, validation_data_size = 1000, epoches = 100, scope_name = 'vars'):
    """
    This function does the training using default or user-defined paremeters
    sess:             The tensorflow session
    train_size:       the # of batches used in the training
    batch_size:       the # of samples in each batch
    validation_data_size: the # of samples used in validation
    epoches:          epoches of training
    scope_name:       the variable scope for trainable variables
    """
    inputs = tf.placeholder(tf.float32, shape = (None, 200, 200, 1))
    outputs = tf.placeholder(tf.float32, shape = (None, 3))
    train_loss, logits, train_op = ConvNet(inputs, outputs)

    # save the checkpoints
    saved_variables = tf.global_variables()
    saver = tf.train.Saver(saved_variables)

    #variables initialization
    sess.run(tf.global_variables_initializer())

    # model training/validation, save the best-validated checkpoint
    maximal_accu = -1
    for epoch in range(epoches):
            # training steps
            losses = []
            for batch in range(training_size):
                    # generate a batch of samples
                    for idx in range(batch_size):
                           params, img = noisy_circle(200, 50, 2)
                           img = np.reshape(img, (1, 200, 200, 1))
                           params = np.reshape(np.array(params), (1, 3))
                           if idx == 0:
                                    X = img
                                    Y = params
                           else:
                                    X = np.concatenate((X, img), axis = 0)
                                    Y = np.concatenate((Y, params), axis = 0)
                    with tf.variable_scope(scope_name, reuse = True):
                           loss, predicts, _ = sess.run([train_loss, logits, train_op], feed_dict = {inputs: X, outputs: Y})
                    losses.append(loss)
            print('The training loss at epoch: ' + str(epoch) + ' is: ' + str(np.mean(np.array(losses))))
        
            # validation steps
            predictions = ConvNet(inputs, outputs, mode = 'predict')
            ious = []
            for batch in range(validation_data_size):
                    params, img = noisy_circle(200, 50, 2)
                    img = np.reshape(img, (1, 200, 200, 1))
                    X = img
                    Y = np.reshape(np.array(params), (1,3))
                    with tf.variable_scope(scope_name, reuse = True):
                            predict = sess.run(predictions, feed_dict = {inputs: X, outputs: Y})
                            row0, col0, rad0 = np.squeeze(Y).tolist()
                            row1, col1, rad1 = np.squeeze(predict).tolist()
                            shape0 = Point(row0, col0).buffer(rad0)
                            shape1 = Point(row1, col1).buffer(rad1)
                            iou = shape0.intersection(shape1).area / shape0.union(shape1).area
                    ious.append(iou)
        
            validation_accu = (np.array(ious) > 0.7).mean()
            print('The validation accuracy at epoch: ' + str(epoch) + ' is: ' + str(validation_accu))

            if validation_accu >= maximal_accu:
                    maximal_accu = validation_accu
                    print('----------------------------------------Better checkpoint found at epoch: ' + str(epoch))
                    os.system('rm checkpoints/*')
                    saver.save(sess, 'checkpoints/dump', global_step=epoch)



