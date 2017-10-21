# coding utf-8

import cfg
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from ml import ClassifierManager
from data_helpers import DataHelper, Words, load_csv
from word2vec import W2VModelManager
from doc2vec import D2VModelManager
from tfidf_chi import CHI, IDF, TFIDF

# 预处理
# dh = DataHelper()
# train_df, test_df = dh.clean_df()


# 分词 -> train_words.csv test_words.csv
# words = Words()
# words.set_user_dict()
# words.set_stop_words()
# jieba.enable_parallel(4)  # 并行分词
# words.cut_sens([train_df, test_df])  # 分词


# CHI -> chi.txt
# chi = CHI()
# chi.run()


# IDF -> idf.txt
# idf = IDF()
# idf.compute_idf()


# 提取关键词 -> trian_tags_pos.csv train_tags_neg.csv
# tfidf = TFIDF()
# tfidf.set_idf_file('idf.txt')
# tfidf.set_stop_words('stop_words.txt')
# # tfidf.extract_train_tags(head_topK=4, content_topK=60, TF=True, withWeight=False)
# tfidf.extract_tags_all(head_topK=6, content_topK=100, TF=True, withWeight=False)


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
w2v = w2vm.load_model('sg.w2v')
X = []
y = []
train_tags_df = load_csv(cfg.DATA_PATH + 'train_tags.csv')
for n in range(5000):
    tags = []
    tags.extend(train_tags_df.iloc[n]['head'].split())
    tags.extend(train_tags_df.iloc[n]['content'].split())
    for tag in tags:
        try:
            X.extend(w2v[tag])
        except:
            X.extend([0]*200)
    if train_tags_df.iloc[n]['label'] == 'POSITIVE':
        y.append(1)
    else:
        y.append(0)
X_train = np.array(X).reshape(-1, 106, 200, 1)
y_train = np.array(y)
print(X_train.shape)
print(y_train.shape)
textcnn.run(X_train, y_train)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# y_train = train_tags_df.iloc[0:100]['label'].replace(['NEGATIVE', 'POSITIVE'], [0, 1])
# temp = np.array([X_train, y])
# temp = temp.transpose()
# np.random.shuffle(temp)

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


# CNN
# cnn.run(X_train, y_train)


