# -- coding: utf-8 --

import scipy.io
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from data_Houston import patch_size, prepare_data, num_band
import time
import os
import scipy.ndimage


# 神经网络参数
num_classes = 15
Train_Batch_Size = 200
Learning_Rate_Base = 0.1
Training_Steps = 1401


def train():
    Training_Data, Testing_Data, Training_Label, Testing_Label, All_Patches, All_Labels = prepare_data()
    test_ind = {}
    test_ind['Training_Data'] = Training_Data
    scipy.io.savemat(os.path.join(os.getcwd(), 'data/Training_Data'), test_ind)
    test_ind = {}
    test_ind['Testing_Data'] = Testing_Data
    scipy.io.savemat(os.path.join(os.getcwd(), 'data/Testing_Data'), test_ind)
    test_ind = {}
    test_ind['Training_Label'] = Training_Label
    scipy.io.savemat(os.path.join(os.getcwd(), 'data/Training_Label'), test_ind)
    test_ind = {}
    test_ind['Testing_Label'] = Testing_Label
    scipy.io.savemat(os.path.join(os.getcwd(), 'data/Testing_Label'), test_ind)
    test_ind = {}
    test_ind['All_Patches'] = All_Patches
    scipy.io.savemat(os.path.join(os.getcwd(), 'data/All_Patches'), test_ind)
    test_ind = {}
    test_ind['All_Labels'] = All_Labels
    scipy.io.savemat(os.path.join(os.getcwd(),'data/All_Labels'), test_ind)


    num_train = Training_Data.shape[0]
    num_test = Testing_Data.shape[0]
    num_total = All_Patches.shape[0]
    x = tf.placeholder(tf.float32, [None, patch_size, patch_size, num_band], name='x_input')
    y = tf.placeholder(tf.float32, [None, num_classes], name='y_input')
    training_flag = tf.placeholder(tf.bool)

    # conv1
    weights1 = tf.get_variable("weigts1", shape=[3, 3, num_band, 96],
                               dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    conv1 = tf.nn.conv2d(x, weights1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.layers.batch_normalization(conv1, training=training_flag)
    conv1 = tf.nn.relu(conv1)

    # conv2
    weights2 = tf.get_variable("weigts2", shape=[3, 3, 96, 96],
                               dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    conv2 = tf.nn.conv2d(conv1, weights2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.layers.batch_normalization(conv2, training=training_flag)
    conv2 = tf.nn.relu(conv2)

    pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv3
    weights3 = tf.get_variable("weigts3", shape=[3, 3, 96, 108],
                               dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    conv3 = tf.nn.conv2d(pool1, weights3, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.layers.batch_normalization(conv3, training=training_flag)
    conv3 = tf.nn.relu(conv3)

    # conv4
    weights4 = tf.get_variable("weigts4", shape=[3, 3, 108, 108],
                               dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    conv4 = tf.nn.conv2d(conv3, weights4, strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.layers.batch_normalization(conv4, training=training_flag)
    conv4 = tf.nn.relu(conv4)

    pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv5
    weights5 = tf.get_variable("weigts5", shape=[3, 3, 108, 128],
                               dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    conv5 = tf.nn.conv2d(pool2, weights5, strides=[1, 1, 1, 1], padding='SAME')
    conv5 = tf.layers.batch_normalization(conv5, training=training_flag)
    conv5 = tf.nn.relu(conv5)

    # conv6
    weights6 = tf.get_variable("weigts6", shape=[3, 3, 128, 128],
                               dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    conv6 = tf.nn.conv2d(conv5, weights6, strides=[1, 1, 1, 1], padding='SAME')
    conv6 = tf.layers.batch_normalization(conv6, training=training_flag)
    conv6 = tf.nn.relu(conv6)

    net = slim.avg_pool2d(conv6, 7, padding='VALID')

    net = slim.flatten(net)

    # fc1
    weights7 = tf.get_variable("weigts7", shape=[128, 200],
                               dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    fc1 = tf.matmul(net, weights7)
    fc1 = tf.layers.batch_normalization(fc1, training=training_flag)
    fc1 = tf.nn.relu(fc1)

    # dropout
    net = slim.dropout(fc1, 0.5, is_training=training_flag)

    # fc2
    weights8 = tf.get_variable("weigts8", shape=[200, num_classes],
                               dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    biases8 = tf.get_variable("biases8", shape=[num_classes],
                             dtype=tf.float32, initializer=tf.zeros_initializer())
    pred = tf.matmul(net, weights8) + biases8

    output = tf.argmax(pred, 1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)

    loss = tf.reduce_mean(cross_entropy)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        Learning_Rate_Base, global_step,
        700, 0.25)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)
    # Define accuracy
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.initialize_all_variables()

    saver = tf.train.Saver({'weights1':weights1,'weights2':weights2,'weights3':weights3,
                            'weights4': weights4,'weights5': weights5,'weights6': weights6,
                            'weights7': weights7,'weights8': weights8,'biases8': biases8})


    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess, os.path.join(os.getcwd(), "model/cnn.ckpt"))
        print("save ok")
        for i in range(Training_Steps):
            idx = np.random.choice(num_train, size=Train_Batch_Size, replace=False)
            batch_x = Training_Data[idx, :]
            batch_y = Training_Label[idx, :]
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y, training_flag: True})


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
