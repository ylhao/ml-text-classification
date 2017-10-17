from cfg import *
import pandas as pd


def ml_predict(X, model, test_df, result_name):
    """
    预测
    :param X: 预测集向量 
    :param model: 模型
    :param test_df: 预测集 DataFrame
    :param result_name: 结果名
    """
    y_pred = pd.DataFrame(model.predict(X)).replace([0, 1], ['NEGATIVE', 'POSITIVE'])
    id_test = test_df.iloc[:]['id']
    # 横向合并
    res = pd.concat([id_test, y_pred], axis=1)
    res[res.iloc[:, 1] != 'NEGATIVE'].to_csv(result_path + result_name,
                                             sep=',', header=None, index=None, encoding='utf8')


