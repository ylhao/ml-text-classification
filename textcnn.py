import tensorflow as tf
import numpy as np
import cfg
import os

NUM_CLASSES = 2
LEARNING_RATE = 1e-3
MAX_STEP = 150000
FILTER_SIZE = [3, 4, 5]
DROPOUT_KEEP_PROB = 0.5
BATCH_SIZE = 64
L2_REG_LAMBDA = 0
SEQUENTE_LENGTH = 69
WORD_VECTOR_LENGTH = 200
NUM_FILTERS = 128


def text_cnn(X, sequence_length, num_classes, filter_sizes, word_vector_length, num_filters):

    l2_loss = tf.constant(0.0)
    pooled_outputs = []
    # 为每个 filter 创建一个卷积层和一个池化层
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope('conv-maxpool-%s' % filter_size):
            # 卷积层
            filter_shape = [filter_size, word_vector_length, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            conv = tf.nn.conv2d(
                X,
                W,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            # pooling
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool')
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # dropout
    with tf.name_scope('dropout'):
        h_drop = tf.nn.dropout(h_pool_flat, DROPOUT_KEEP_PROB)

    # output
    with tf.name_scope('output'):
        W = tf.get_variable(
            'W',
            shape=[num_filters_total, num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        scores = tf.nn.xw_plus_b(h_drop, W, b, name='scores')
        return l2_loss, scores


def losses(l2_loss, scores, labels, l2_reg_lambda=L2_REG_LAMBDA):
    with tf.variable_scope('loss') as scope:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=scores,
            labels=labels,
            name='cross_entropy_per_example'
        )
        loss = tf.reduce_mean(loss, name='loss') + l2_reg_lambda * l2_loss
    return loss


def accuracy(scores, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(scores, labels, 1)
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


def optimize(loss):
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)
    return train_op
    # with tf.variable_scope('optimizer') as scope:
    #     train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    # return train_op


def run():

    images, labels = generate_data()
    image_batch, label_batch = set_batch(images, labels)

    l2_loss, scores = text_cnn(
        image_batch,
        sequence_length=SEQUENTE_LENGTH,
        num_classes=NUM_CLASSES,
        filter_sizes=FILTER_SIZE,
        word_vector_length=WORD_VECTOR_LENGTH,
        num_filters=NUM_FILTERS
    )

    train_losses = losses(l2_loss, scores, label_batch, L2_REG_LAMBDA)
    train_op = optimize(train_losses)
    train_acc = accuracy(scores, label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(cfg.LOG_PATH_TEXTCNN, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_losses, train_acc])
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            if step % 50 == 0:
                # print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(cfg.LOG_PATH_TEXTCNN, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


# def generate_data():
#     num = 1000
#     labels = np.random.randint(0, 2, num)
#     images = np.random.random([num, 69, 200, 1])
#     print('label size :{}, image size {}'.format(labels.shape, images.shape))
#     return images, labels

# def generate_data(X, y):
#     # num = 1000
#     # labels = np.random.randint(0, 2, num)
#     # images = np.random.random([num, 784])
#     # images = images.reshape([-1,28,28,1])
#     # print('label size :{}, image size {}'.format(labels.shape, images.shape))
#     # return images, labels
#     labels = y
#     images = X.reshape([-1, 10, 30, 1])
#     return images, labels

run()