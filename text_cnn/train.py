# coding: utf-8

import datetime
import os
import numpy as np
import tensorflow as tf
import time
import cfg
from word2vec import W2VModelManager
from data_helpers import load_csv
from source.text_cnn.text_cnn import TextCNN

# 参数
tf.flags.DEFINE_float('dev_sample_percentage', .2, '验证集比例')
tf.flags.DEFINE_string('train_words_file', cfg.DATA_PATH + 'train_new.csv', '训练集词列表文件')
tf.flags.DEFINE_string('w2v_model', cfg.MODEL_PATH + 'word2vec_model/' + 'sg.w2v', 'word2vec 模型')

tf.flags.DEFINE_integer('num_classes', 2, '类别数')
tf.flags.DEFINE_integer('sequence_length', 400, '每篇文章的词数')
tf.flags.DEFINE_integer('embedding_size', 200, '词向量维度')
tf.flags.DEFINE_string('filter_sizes', '3,4,5', "过滤器尺寸")
tf.flags.DEFINE_integer('num_filters', 128, '每种尺寸过滤器数目')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.flags.DEFINE_float('l2_reg_lambda', 0.0, 'L2 regularization lambda')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')

tf.flags.DEFINE_integer('batch_size', 64, 'Train Batch Size')
tf.flags.DEFINE_integer('evl_batch_size', 64, 'Evl Batch Size')
tf.flags.DEFINE_integer('num_epochs', 200, 'epoch')
tf.flags.DEFINE_integer('evaluate_every', 100, '评测模型的步数节点')
tf.flags.DEFINE_integer('checkpoint_every', 100, '保存模型的步数节点')
tf.flags.DEFINE_integer('num_checkpoints', 5, 'Number of checkpoints to store')

# Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
print('='*120)


words_df = load_csv(FLAGS.train_words_file)[:10000]
words_df = words_df.sample(frac=1)  # 打乱
TRAIN_WORDS_DF = words_df[0:int(words_df.shape[0] * (1 - FLAGS.dev_sample_percentage))]
EVL_WORDS_DF = words_df[int(words_df.shape[0] * (1 - FLAGS.dev_sample_percentage)):]
print('训练集和验证集总样例数：', words_df.shape[0])
print('训练集样例数：', TRAIN_WORDS_DF.shape[0])
print('测试集样例数：', EVL_WORDS_DF.shape[0])
print('='*120)

w2vm = W2VModelManager()
w2v = w2vm.load_model(FLAGS.w2v_model)
print('word2vec 模型信息：', w2v)


def train_batch_iter(train_words_df):
    data_size = train_words_df.shape[0]
    num_batches_per_epoch = int((data_size - 1) / FLAGS.batch_size) + 1
    print('训练集样例数：', data_size)
    print('训练集一个 epoch 的 batch 数：', num_batches_per_epoch)
    for epoch in range(FLAGS.num_epochs):
        # 每个 epoch 都重新打乱数据
        train_words_df = train_words_df.sample(frac=1)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * FLAGS.batch_size
            end_index = min((batch_num + 1) * FLAGS.batch_size, data_size)
            X = []
            y = []
            for n in range(start_index, end_index):
                words = []
                word_num = 0
                words.extend(train_words_df.iloc[n]['content'].split())
                for word in words:
                    try:
                        X.extend(w2v[word])
                    except:
                        X.extend([0] * FLAGS.embedding_size)
                    word_num += 1
                    if word_num == FLAGS.sequence_length:
                        break
                X.extend([0] * (FLAGS.embedding_size * (FLAGS.sequence_length - word_num)))
                if train_words_df.iloc[n]['label'] == 'POSITIVE':
                    y.append(1)
                    y.append(0)
                else:
                    y.append(0)
                    y.append(1)
            X_train = np.array(X).reshape(-1, FLAGS.sequence_length, FLAGS.embedding_size, 1)
            y_train = np.array(y).reshape(-1, 2)
            yield X_train, y_train


def evl_batch_iter(evl_words_df):
    data_size = evl_words_df.shape[0]
    num_batches = int((data_size - 1) / FLAGS.evl_batch_size) + 1
    print('验证集样例数:', data_size)
    print('验证集 batch 数:', num_batches)
    evl_words_df = evl_words_df.sample(frac=1)
    for batch_num in range(num_batches):
        start_index = batch_num * FLAGS.evl_batch_size
        end_index = min((batch_num + 1) * FLAGS.evl_batch_size, data_size)
        X = []
        y = []
        for n in range(start_index, end_index):
            words = []
            word_num = 0
            words.extend(evl_words_df.iloc[n]['content'].split())
            for word in words:
                try:
                    X.extend(w2v[word])
                except:
                    X.extend([0] * FLAGS.embedding_size)
                word_num += 1
                if word_num == FLAGS.sequence_length:
                    break
            X.extend([0] * (FLAGS.embedding_size * (FLAGS.sequence_length - word_num)))
            if evl_words_df.iloc[n]['label'] == 'POSITIVE':
                y.append(1)
                y.append(0)
            else:
                y.append(0)
                y.append(1)
        X_evl = np.array(X).reshape(-1, FLAGS.sequence_length, FLAGS.embedding_size, 1)
        y_evl = np.array(y).reshape(-1, 2)
        yield X_evl, y_evl

# 训练
# ====================================================================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=FLAGS.sequence_length,
            num_classes=FLAGS.num_classes,
            embedding_size=FLAGS.embedding_size,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # 定义训练过程
        global_step = tf.Variable(0, name='global_step', trainable=False)  # 总的训练步数
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name), g)
                sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        # out_dir = os.path.abspath(os.path.join(cfg.TEXT_CNN_PATH, 'runs', timestamp))
        out_dir = cfg.TEXT_CNN_PATH
        print('Writing to {}\n'.format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar('loss', cnn.loss)
        acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # 检查文件夹存不存在，不存在则创建文件夹
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # 初始化变量
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict,
            )
            # step = sess.run(train_op)
            time_str = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0

            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        batches = train_batch_iter(TRAIN_WORDS_DF)
        for batch in batches:
            x_batch, y_batch = batch[0], batch[1]
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:  # 判断是否需要验证模型
                print('\nEvaluation:')
                for batch in evl_batch_iter(EVL_WORDS_DF):
                    dev_step(batch[0], batch[1], writer=dev_summary_writer)
                print('')
            if current_step % FLAGS.checkpoint_every == 0:  # 判断是否需要保存
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print('Saved model checkpoint to {}\n'.format(path))
