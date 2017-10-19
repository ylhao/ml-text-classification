# coding: utf-8

import cfg
from nlp import load_csv
import numpy as np
from ml import ClassifierManager

train_df = load_csv(cfg.DATA_PATH + 'train_words.csv')

X = []
y = []

words_dict = {}
with open(cfg.DATA_PATH + 'chi.txt', 'r') as f:
    i = 0
    while i < 2000:
        words_dict[f.readline().strip()] = 0
        i += 1
print(len(words_dict))

for i in range(train_df.shape[0]):
# for i in range(200):

    words = []

    try:
        words.extend(train_df.iloc[i]['head'].split())
    except Exception:
        pass

    try:
        words.extend(train_df.iloc[i]['content'].split())
    except Exception:
        pass

    for w in words_dict:
        words_dict[w] = 0

    for w in words:
        if w in words_dict:
            words_dict[w] = 1.0

    for w in words_dict:
        X.append(words_dict[w])

    if train_df.iloc[i]['label'] == 'POSITIVE':
        y.append(1)
    else:
        y.append(0)

    if i % 10000 == 0:
        print(i)


X_train = np.array(X).reshape(train_df.shape[0], 2000)
print(X_train.shape)
y_train = np.array(y)
print(y_train.shape)


# ----------------------------------------------------------------------------------

# 机器学习 评估分类器 训练分类器 获取预测结果
cm = ClassifierManager(X_train, y_train)
cm.run()

# for classifier in cm.classifiers:
#     cm.get_predict_result(X_test, classifier, test_df, 'ml_'+classifier.classifier_name.split('.')[0]+'.csv')

