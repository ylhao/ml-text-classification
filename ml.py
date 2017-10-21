# coding:utf-8

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import pickle
import cfg


class Classifier:
    def __init__(self, X, y, num, classifier_name):
        """
        分类器基类
        :param X: X_train 
        :param y: y_train
        :param num: 寻参样本数目 
        """
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X[:num], y[:num], test_size=0.2)
        self.classifier_name = classifier_name
        self.classifier = None  # 需在子类中重新定义
        self.param_grid = None  # 如果需要，可在子类中重新定义
        self.best_params_ = None  # 与原生参数命名保持一致

    def grid_search(self):
        """
        网格寻参，将找到的最优参数保存为 self.best_params_，寻参结束自动用最优参数对分类器进行评估
        """
        clf = GridSearchCV(self.classifier, self.param_grid, scoring='f1', n_jobs=10, cv=5)
        clf.fit(self.X_train, self.y_train)
        self.best_params_ = clf.best_params_
        print(self.best_params_)  # 输出最优参数
        y_pred = clf.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))

    def evalue_classifier(self):
        """
        评估模型
        """
        self.classifier.fit(self.X_train, self.y_train)
        y_pred = self.classifier.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))

    def redefine_classifier(self):
        """
        redefine classifier
        修改 classifier 的参数为最优参数
        """
        pass

    def train_classifier(self):
        """
        train classifier
        """
        self.classifier.fit(self.X, self.y)

    def save_classifier(self):
        """
        save classifier to disk
        """
        with open(cfg.CLASSIFIER_PATH + self.classifier_name, 'wb') as f:
            pickle.dump(self.classifier, f)
        print('save %s done' % self.classifier_name)

    def print_classifier(self):
        """
        打印分类器的信息
        """
        print(self.classifier)

    def predict(self, X_test):
        """
        预测
        :param X_test: 测试集
        :return: 预测结果 DataFrame
        """
        y_pred = pd.DataFrame(self.classifier.predict(X_test)).replace([0, 1], ['NEGATIVE', 'POSITIVE'])
        return y_pred

    def run(self):
        # 如果定义了寻参范围，那么说明需要进行网格搜索，寻找最优参数
        if self.param_grid:
            self.grid_search()  # 寻参结束后，会自动评估分类器
            self.redefine_classifier()
        else:
            self.evalue_classifier()
        # self.train_classifier()  # 训练分类器
        # self.print_classifier()  # 打印分类器
        # self.save_classifier()  # 保存分类器


class LRClassifier(Classifier):
    """
    逻辑回归分类器
    """
    def __init__(self, X, y, num, classifier_name):
        super().__init__(X, y, num, classifier_name)
        self.classifier = LogisticRegression(n_jobs=-1)


class DTClassifier(Classifier):
    """
    决策树分类器
    """
    def __init__(self, X, y, num, classifier_name):
        super().__init__(X, y, num, classifier_name)
        self.classifier = DecisionTreeClassifier()


class MNBClassifier(Classifier):
    """
    朴素贝叶斯分类器
    """
    def __init__(self, X, y, num, classifier_name):
        super().__init__(X, y, num, classifier_name)
        self.classifier = MultinomialNB()


class SVCClassifier(Classifier):
    """
    SVM 分类器
    """
    def __init__(self, X, y, num, classifier_name):
        super().__init__(X, y, num, classifier_name)
        self.classifier = SVC(kernel='rbf', class_weight='balanced')
        self.param_grid = {
            'C': [2 ** i for i in range(-5, 15, 2)],
            'gamma': [2 ** i for i in range(3, -15, -2)],
        }

    def redefine_classifier(self):
        self.classifier = SVC(kernel='rbf', class_weight='balanced',
                  C=self.best_params_['C'], gamma=self.best_params_['gamma'])


class LinearClassifier(Classifier):
    """
    线性 SVM 分类器
    """
    def __init__(self, X, y, num, classifier_name):
        super().__init__(X, y, num, classifier_name)
        self.classifier = LinearSVC(class_weight='balanced')
        self.param_grid = {'C': [2 ** i for i in range(-5, 15, 2)], }

    def redefine_classifier(self):
        self.classifier = LinearSVC(class_weight='balanced', C=self.best_params_['C'])


class RFClassifier(Classifier):
    """
    随机森林分类器
    """
    def __init__(self, X, y, num, classifier_name):
        super().__init__(X, y, num, classifier_name)
        self.classifier = RandomForestClassifier(n_jobs=-1)
        self.param_grid = {'n_estimators': [1, 5, 10, 50, 100, 200, 500, 1000], }

    def redefine_classifier(self):
        self.classifier = RandomForestClassifier(n_jobs=-1, n_estimators=self.best_params_['n_estimators'])


class KNNClassifier(Classifier):
    """
    KNN 分类器
    """
    def __init__(self, X, y, num, classifier_name):
        super().__init__(X, y, num, classifier_name)
        self.classifier = KNeighborsClassifier(n_jobs=-1)
        self.param_grid = {'n_neighbors' : np.arange(1,32,1), }

    def redefine_classifier(self):
        self.classifier = KNeighborsClassifier(n_jobs=-1, n_neighbors=self.best_params_['n_neighbors'])


class ClassifierManager:
    """
    分类器管理类
    """
    def __init__(self, X, y):
        self.classifiers = [
            # RFClassifier(X, y, 20000, 'rf.model'),
            LinearClassifier(X, y, 20000, 'linear.model'),
            # MNBClassifier(X, y, 20000, 'mnb.model'),
            # LRClassifier(X, y, 20000, 'lr.model'),
            # KNNClassifier(X, y, 20000, 'knn.model'),
            # SVCClassifier(X, y, 20000, 'svc.model'),
        ]

    def run(self):
        for classifier in self.classifiers:
            classifier.run()

    @staticmethod
    def get_predict_result(X, classifier, test_df, result_name):
        """
        预测
        :param X: X_test
        :param classifier: classifier
        :param test_df: 预测集 DataFrame
        :param result_name: 结果文件名
        """
        y_pred = pd.DataFrame(classifier.predict(X)).replace([0, 1], ['NEGATIVE', 'POSITIVE'])
        id_test = test_df.iloc[:]['id']
        # 横向合并
        res = pd.concat([id_test, y_pred], axis=1)
        res[res.iloc[:, 1] != 'NEGATIVE'].to_csv(cfg.RESULT_PATH + result_name,
                                                 sep=',', header=None, index=None, encoding='utf8')