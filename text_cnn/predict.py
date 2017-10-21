# coding: utf-8

import csv
import os
import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string('test_tags_file', 'test_tags.csv', 'Data source for the testing set')
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
print("\nEvaluating...\n")




def evl_batch_iter(evl_tags_df):
    data_size = evl_tags_df.shape[0]  # 验证集样例数
    print('验证集样例数:', data_size)
    num_batches = int((data_size - 1) / FLAGS.batch_size) + 1
    print('验证集一个 epoch 的 batch 数:', num_batches)
    for batch_num in range(num_batches):  # batch
        start_index = batch_num * FLAGS.batch_size
        end_index = min((batch_num + 1) * FLAGS.batch_size, data_size)
        X = []
        y = []
        id = []
        for n in range(start_index, end_index):
            tags = []
            tags.extend(evl_tags_df.iloc[n]['head'].split())
            tags.extend(evl_tags_df.iloc[n]['content'].split())
            for tag in tags:
                try:
                    X.extend(w2v[tag])
                except:
                    X.extend([0] * FLAGS.embedding_size)
            if tags_df.iloc[n]['label'] == 'POSITIVE':  # POSITIVE [1, 0]
                y.append(1)
                y.append(0)
            else:  # NEGATIVE [0, 1]
                y.append(0)
                y.append(1)
            id.append(tags_df.iloc[n]['id'])
        X_evl = np.array(X).reshape(-1, FLAGS.sequence_length, FLAGS.embedding_size, 1)
        y_evl = np.array(y).reshape(-1, 2)
        # 小范围打乱数据 首先根据 X1 的尺寸得到新的一个随机顺序
        ri = np.random.permutation(len(y_evl))
        X_evl = X_evl[ri]
        y_evl = y_evl[ri]
        id = np.array(id)
        id = id[ri]
        ###################################################
        # 数据缩放
        ###################################################
        yield X_evl, y_evl, id
# Evaluation
# ====================================================================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]

        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        evl_batches = evl_batch_iter(evl_tags_df)

        # Collect the predictions here
        all_predictions = []
        labels = []
        ids = []

        for evl_batch in evl_batches:
            X_evl = evl_batch[0]
            y_evl = evl_batch[1]
            id_evl = evl_batch[2]
            batch_predictions = sess.run(predictions, {input_x: X_evl, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            labels = np.concatenate([labels, y_evl])
            ids = np.concatenate([ids, id_evl])

    correct_predictions = float(sum(all_predictions == labels))
    print("Total number of test examples: {}".format(len(labels)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(labels))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(ids), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
