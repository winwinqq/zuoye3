import re
import numpy as np
from jieba import cut
from itertools import chain
from imblearn.over_sampling import SMOTE  # 新增导入
from collections import Counter
from sklearn.naive_bayes import MultinomialNB

def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            line = cut(line)
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words

def get_top_words(top_num):
    """获取高频特征词"""
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    all_words = []
    for filename in filename_list:
        all_words.append(get_words(filename))
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]

# 特征工程
top_words = get_top_words(100)
vector = []
for words in [get_words(f'邮件_files/{i}.txt') for i in range(151)]:
    word_map = list(map(lambda word: words.count(word), top_words))
    vector.append(word_map)
vector = np.array(vector)
labels = np.array([1]*127 + [0]*24)

# 新增SMOTE过采样处理
sm = SMOTE(random_state=42)
vector_res, labels_res = sm.fit_resample(vector, labels)

# 模型训练
model = MultinomialNB()
model.fit(vector_res, labels_res)  # 使用平衡后的数据训练

def predict(filename):
    """预测函数保持不变"""
    words = get_words(filename)
    current_vector = np.array(tuple(map(lambda word: words.count(word), top_words)))
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'

# 测试输出保持不变
print('151.txt分类情况:{}'.format(predict('邮件_files/151.txt')))
print('152.txt分类情况:{}'.format(predict('邮件_files/152.txt')))
print('153.txt分类情况:{}'.format(predict('邮件_files/153.txt')))
print('154.txt分类情况:{}'.format(predict('邮件_files/154.txt')))
print('155.txt分类情况:{}'.format(predict('邮件_files/155.txt')))