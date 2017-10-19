# coding utf-8

import pandas as pd
import numpy as np
import codecs
import cfg


class Analyse:

    def __init__(self):
        self.train_file = cfg.DATA_PATH + 'train.tsv'
        self.test_file = cfg.DATA_PATH + 'evaluation_public.tsv'
        # 数据框
        self.train_df = pd.read_csv(self.train_file, sep='\t|\n', encoding='utf8', header=None,
                                    names=['id', 'head', 'content', 'label'], engine='python')
        self.test_df = pd.read_csv(self.test_file, sep='\t|\n', encoding='utf8',
                                   header=None, names=['id', 'head', 'content'], engine='python')

    def rule(self):
        pos_df = self.train_df[self.train_df['label'] == 'POSITIVE']
        neg_df = self.train_df[self.train_df['label'] == 'NEGATIVE']
        print(pos_df.shape)
        print(neg_df.shape)
        combine = [pos_df, neg_df]

        for df in combine:
            lens = []
            for line_num in range(df.shape[0]):
                content = df.iloc[line_num]['content']
                lens.append(len(content))
            lens = np.array(lens)
            lens_df = pd.DataFrame(lens, columns=['slen'])
            df = pd.concat([df, lens_df], axis=1)
            print(df.describe())


analyse = Analyse()
analyse.rule()

