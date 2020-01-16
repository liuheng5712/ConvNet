import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt

from models import ConvNet
from utils import Trainer
import tensorflow as tf

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


def find_circle(img, sess, inputs, outputs, predictions):
    # Fill in this function
    # inference function
    # reshape the image if not in 4 dim
    scope_name = 'vars'
    if len(img.shape) < 4:
            X = np.reshape(img, (1, 200, 200, 1))
    # a dumpy variable Y
    Y = np.reshape(np.array([0,0,0]), (1,3))
    with tf.variable_scope(scope_name, reuse = True):
            predict = sess.run(predictions, feed_dict = {inputs: X, outputs: Y})
    
    return np.squeeze(predict).tolist()


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1
    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)
    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def main():
    results = []
    # Modification starts
    sess = tf.Session()
    # if we don't have the trained model, simply do:
    # Trainer(sess)
    # pass the session and the image to find_circle function
    checkpoint_path = 'checkpoints/dump-63'
    inputs = tf.placeholder(tf.float32, shape = (None, 200, 200, 1))
    outputs = tf.placeholder(tf.float32, shape = (None, 3))
    predictions = ConvNet(inputs, outputs, mode = 'predict')

    saved_variables = tf.global_variables()
    saver = tf.train.Saver(saved_variables)
    saver.restore(sess, checkpoint_path)
    # End of modification
    for idx in range(1000):
        print('Inference on image: ' + str(idx))
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(img, sess, inputs, outputs, predictions)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())

    sess.close()

    
