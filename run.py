# coding utf-8

from doc2vec import D2VModelManager
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import cfg
import numpy as np
from ml import ClassifierManager
from nlp import DataHelper
from nlp import Doc2Words
import cnn
from word2vec import W2VModelManager
import pandas as pd

# ----------------------------------------------------------------------------------

# 预处理
p = DataHelper()
train_df, test_df = p.run()

# ----------------------------------------------------------------------------------

# 分词
# save test_words.csv and train_words.csv to disk
doc2words = Doc2Words()
doc2words.run([train_df, test_df])

# ----------------------------------------------------------------------------------

# 训练文档向量
# save doc2vec model to disk
# d2vm = D2VModelManager()
# d2v = d2vm.train_model()

# ---------------------------------------------------------------------------------

# 训练词向量
# save word2vec model to disk
w2vm = W2VModelManager()
w2v = w2vm.train_model()

# ---------------------------------------------------------------------------------

# load doc2vec model
# d2vm = D2VModelManager()
# d2v = d2vm.load_model('dm.d2v')

# ---------------------------------------------------------------------------------

# load word2vec model
# w2vm = W2VModelManager()
# w2v = w2vm.load_model('sg.w2v')

# ---------------------------------------------------------------------------------

# get y_train X_train, X_test
# y_train = train_df.iloc[:]['label'].replace(['NEGATIVE', 'POSITIVE'], [0, 1])
# print(y_train.shape)
#
# X = np.array(d2v.docvecs)
#
# # maximum minimum scaling
# scaler = MinMaxScaler()
# scaler.fit(X)
# X = scaler.transform(X)
#
# # standard scaling
# # scaler = StandardScaler()
# # scaler.fit(X)
# # X = scaler.transform(X)
#
# X_train = X[:y_train.shape[0]]
# X_test = X[y_train.shape[0]:]

# ----------------------------------------------------------------------------------

# 机器学习 评估分类器 训练分类器 获取预测结果
# cm = ClassifierManager(X_train, y_train)
# cm.run()

# for classifier in cm.classifiers:
#     cm.get_predict_result(X_test, classifier, test_df, 'ml_'+classifier.classifier_name.split('.')[0]+'.csv')

# ----------------------------------------------------------------------------------

# CNN
# cnn.run(X_train, y_train)


