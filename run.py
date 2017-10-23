# coding utf-8

import cfg
import jieba
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from ml import ClassifierManager
from data_helpers import DataHelper, Words, load_csv
from word2vec import W2VModelManager
from doc2vec import D2VModelManager
from tfidf_chi import CHI, IDF, TFIDF

# 预处理 训练集去空、去重 测试集去空 -> train_df test_df
# dh = DataHelper()
# train_df, test_df = dh.clean_df()


# 分词 -> train_words.csv test_words.csv train_words_clean.csv test_words_clean.csv
words = Words()
# words.set_user_dict()
# words.set_stop_words()
# words.cut_sens([train_df, test_df])  # 分词
# words.clean()


# IDF -> idf.txt
# idf = IDF()
# idf.compute_idf()


# 提取关键词 -> trian_tags_pos.csv train_tags_neg.csv
# tfidf = TFIDF()
# tfidf.set_idf_file(cfg.DATA_PATH + 'idf.txt')
# tfidf.set_stop_words(cfg.DATA_PATH + 'stop_words.txt')
# tfidf.extract_tags_all(head_topK=6, content_topK=200, TF=False, withWeight=False)


# CHI -> chi.txt
# chi = CHI()
# chi.run()


# 训练文档向量 -> dm.d2v
# d2vm = D2VModelManager()
# d2v = d2vm.train_model()


# 训练词向量 -> sg.w2v
# save word2vec model to disk
# w2vm = W2VModelManager()
# w2v = w2vm.train_model()


# 加载 doc2vec 模型
# d2vm = D2VModelManager()
# d2v = d2vm.load_model('dm.d2v')


# 加载 word2vec 模型
w2vm = W2VModelManager()
w2v = w2vm.load_model(cfg.MODEL_PATH + 'sg.w2v')
w2vm.model_test(cfg.MODEL_PATH + 'sg.w2v')

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# y_train = train_tags_df.iloc[0:100]['label'].replace(['NEGATIVE', 'POSITIVE'], [0, 1])

# 提取标签 POSITIVE -> 1 NEGATIVE -> 0
# y_train = train_df.iloc[:]['label'].replace(['NEGATIVE', 'POSITIVE'], [0, 1])
# print('y_train shape:', y_train.shape)
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


# 机器学习 评估分类器 训练分类器 获取预测结果
# cm = ClassifierManager(X_train, y_train)
# cm.run()

# for classifier in cm.classifiers:
#     cm.get_predict_result(X_test, classifier, test_df, 'ml_'+classifier.classifier_name.split('.')[0]+'.csv')
