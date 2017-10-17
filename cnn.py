# coding: utf-8

import tensorflow as tf
import numpy as np
import cfg
import os

BATCH_SIZE = 128
N_CLASSES = 2
LEARNING_RATE = 1e-4
MAX_STEP = 150000


def build_model(images, batch_size, n_classes):
    # 卷积层1
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable(
            'weights',
            shape=[5, 5, 1, 32],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32)
        )
        biases = tf.get_variable(
            'biases',
            shape=[32],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.1)
        )
        conv = tf.nn.conv2d(
            images,
            weights,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # 池化层1
    with tf.variable_scope('pooling1') as scope:
        pool1 = tf.nn.max_pool(
            conv1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='max_pool1'
        )

    # 卷积层2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable(
            'weights',
            shape=[5, 5, 32, 64],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)
        )
        biases = tf.get_variable(
            'biases',
            shape=[64],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.1)
        )
        conv = tf.nn.conv2d(
            pool1,
            weights,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # 池化层2
    with tf.variable_scope('pooling2') as scope:
        pool2 = tf.nn.max_pool(
            conv2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='max_pool2'
        )

    # 全连接层
    with tf.variable_scope('local1') as scope:
        pool2_flat = tf.reshape(pool2, [batch_size, -1])
        dim = pool2_flat.get_shape()[1].value
        weights = tf.get_variable(
            'weights',
            shape=[dim, 1024],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32)
        )
        biases = tf.get_variable(
            'biases',
            shape=[1024],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.1)
        )
        local1 = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases, name=scope.name)

    """
    这里可以继续添加全连接层
    """

    # 输出层
    with tf.variable_scope('softmax') as scope:
        weights = tf.get_variable(
            'softmax',
            shape=[1024, n_classes],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32)
        )
        biases = tf.get_variable(
            'biases',
            shape=[n_classes],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.1))
        softmax = tf.add(tf.matmul(local1, weights), biases, name='softmax')
        print('softmax shape', softmax.shape)
    return softmax


def losses(logits, labels):
    print('labels shape', labels.shape)
    with tf.variable_scope('loss') as scope:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels,
            name='cross_entropy_per_example'
        )
        loss = tf.reduce_mean(loss, name='loss')
    return loss


def optimize(loss):
    with tf.variable_scope('optimizer') as scope:
        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    return train_op


def evaluation(preds, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(preds, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy


def set_batch(images, labels):
    images = tf.cast(images, tf.float32)
    label = tf.cast(labels, tf.int32)
    input_queue = tf.train.slice_input_producer([images, label], shuffle=True)
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=BATCH_SIZE, num_threads=8, capacity=64)
    return image_batch, label_batch


def run(X, y):

    images, labels = generate_data(X, y)
    image_batch, label_batch = set_batch(images, labels)

    train_logits = build_model(image_batch, BATCH_SIZE, 2)
    train_losses = losses(train_logits, label_batch)
    train_op = optimize(train_losses)
    train_acc = evaluation(train_logits, label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(cfg.log_path, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_losses, train_acc])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(cfg.log_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


def generate_data(X, y):
    # num = 1000
    # labels = np.random.randint(0, 2, num)
    # images = np.random.random([num, 784])
    # images = images.reshape([-1,28,28,1])
    # print('label size :{}, image size {}'.format(labels.shape, images.shape))
    # return images, labels
    labels = y
    images = X.reshape([-1, 10, 30, 1])
    return images, labels