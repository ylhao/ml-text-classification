# coding: utf-8

import cfg
import re
import codecs


def pre_train():
    label_dict = {}
    with open(cfg.DATA_PATH + 'evaluation_public_origin.tsv', 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
    # 删除文章内容为空的样例, 所有文章内容为空的样例都判定为负标签
    for n, line in enumerate(lines):
        if line[2] == '':
            label_dict[line[0]] = 'NEGATIVE'
            lines.pop(n)

    # 删除完全相同的文章, 如果两篇文章完全相同, 判定文章为正标签
    content_pool = {}
    for n, line in enumerate(lines):
        if line[2] != '':
            if line[2] not in content_pool:
                content_pool[line[2]] = [n, line[0]]
            else:
                label_dict[line[0]] = 'POSITIVE'
                id_dup = content_pool[line[2]][1]  # 取出与之重复的那篇文章的 id
                line_num = content_pool[line[2]][0]  # 取出与之重复的那篇文章的行号
                label_dict[id_dup] = 'POSITIVE'
                lines.pop(n)
                lines.pop(line_num)

    # 开头是{，接着是两个或两个以上的空白符或字母，接着是是汉字，接着是空白符或者非空白符，结尾为}
    PATTERN = re.compile(u'\{[\sA-Za-z:0-9\-_;\.]{2,}[\u4E00-\u9FA5]+[\s\S]*\}')
    KEYS = ('pgc', 'text-decoration', 'none', 'outline', 'display', 'block', 'width',
           'height', 'solid', 'position', 'relative', 'padding', 'absolute', 'background',
           'top', 'left', 'right', 'cover','font-size', 'font', 'overflow', 'bold',
           'hidden', 'inline', 'block', 'align', 'center', 'transform', 'space', 'vertical',
            'color', 'webkit', 'translatntent')
    for n, line in enumerate(lines):
        key_num = 0
        for key in KEYS:
            if key in line[2]:
                key_num += 1
        if key_num >= 1:
            if re.search(PATTERN, line[2].replace(' ', '')):
                label_dict[line[0]] = 'NEGATIVE'
                lines.pop(n)

    fw = codecs.open(cfg.DATA_PATH + 'evaluation_public.tsv', 'w', encoding='utf8')
    for line in lines:
        fw.write('%s\n' % '\t'.join(line))


def pre_evl():
    label_dict = {}
    with open(cfg.DATA_PATH + 'train_origin.tsv', 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]

    for n, line in enumerate(lines):
        if line[2] == '':
            lines.pop(n)

    content_pool = {}
    for n, line in enumerate(lines):
        if line[2] != '':
            if line[2] not in content_pool:
                content_pool[line[2]] = [n, line[0]]
            else:
                id_dup = content_pool[line[2]][1]
                line_num = content_pool[line[2]][0]
                lines.pop(n)
                lines.pop(line_num)

    # 开头是{，接着是两个或两个以上的空白符或字母，接着是是汉字，接着是空白符或者非空白符，结尾为}
    PATTERN = re.compile(u'\{[\sA-Za-z:0-9\-_;\.]{2,}[\u4E00-\u9FA5]+[\s\S]*\}')
    KEYS = ('pgc', 'text-decoration', 'none', 'outline', 'display', 'block', 'width',
           'height', 'solid', 'position', 'relative', 'padding', 'absolute', 'background',
           'top', 'left', 'right', 'cover','font-size', 'font', 'overflow', 'bold',
           'hidden', 'inline', 'block', 'align', 'center', 'transform', 'space', 'vertical',
            'color', 'webkit', 'translatntent')
    for n, line in enumerate(lines):
        key_num = 0
        for key in KEYS:
            if key in line[2]:
                key_num += 1
        if key_num >= 1:
            if re.search(PATTERN, line[2].replace(' ', '')):
                lines.pop(n)

    fw = codecs.open(cfg.DATA_PATH + 'train.tsv', 'w', encoding='utf8')
    for line in lines:
        fw.write('%s\n' % '\t'.join(line))

    fw = open(cfg.DATA_PATH + 'result_part.csv', 'w')
    for key in label_dict:
        fw.write('%s,%s\n' % (key, label_dict[key]))

