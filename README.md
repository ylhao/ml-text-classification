# java_hello_world

### 问题
1. ovo 与 ovr
2. pd.read_csv()　的参数 sep 为什么设置成 \n|\t
3. df['h_c'] = df['headline'] + df['content']
4. 停用词表里面第一个停用词是空格，不知道是否必要这样做
5. epochs=1设置是否合理需要研究一下

### 软件包
- jieba
- codecs
- gensim
- numpy
- pandas
- sklearn
- tensorflow
- keras
- pickle

### 软件包安装

```bash
sudo apt update
sudo apt-get install python3-pip
sudo pip3 install jieba
sudo pip3 install numpy
sudo pip3 install pandas

sudo apt-get install libpng-dev
wget http://ftp.yzu.edu.tw/nongnu//freetype/freetype-2.8.1.tar.gz
tar zxvf freetype-2.6.tar.gz
./configure 
make
sudo apt install python-dev
sudo apt-get build-dep python-scipy
sudo pip3 install scipy

sudo pip3 install -U scikit-learn
sudo pip3 install --upgrade gensim
sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl
```
### git 操作

```bash
git init
git remote add origin https://github.com/ilhao/java_test
git config user.name ""
git config user.email ""
git pull origin master
git push origin master
```

### 代码速度的处理
- 2VModelManager: 　Doc2Vec，12个线程
- W2VModelManager: Word2Vec, 12个线程
- Doc2Words: 用了 2 * 12个线程

### 参考资料
L0,L1,L2范数 http://blog.csdn.net/zouxy09/article/details/24971995


### 对结巴分词的修改
提取关键词，按原顺序返回

```python
def my_extract_tags(self, words, topK=20, withWeight=False, allowPOS=(), withFlag=False):
    """
    自定义的提取关键词的函数
        - words: 词列表 []
    """
    freq = {}
    for w in words:
        if allowPOS:
            if w.flag not in allowPOS:
                continue
            elif not withFlag:
                w = w.word
        wc = w.word if allowPOS and withFlag else w
        if len(wc.strip()) < 2 or wc.lower() in self.stop_words:
            continue
        freq[w] = freq.get(w, 0.0) + 1.0
    total = sum(freq.values())
    for k in freq:
        kw = k.word if allowPOS and withFlag else k
        freq[k] *= self.idf_freq.get(kw, self.median_idf) / total
    tags = sorted(freq, key=freq.__getitem__, reverse=True)
    tags = tuple(tags[:topK])
    tags1 = []
    for w in words:
        if w in tags and w not in tags1:
            tags1.append(word)
    return tags1
```
