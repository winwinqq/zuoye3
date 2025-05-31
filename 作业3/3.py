import re
import numpy as np
from jieba import cut
from itertools import chain
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split  # 新增
from sklearn.metrics import classification_report  # 新增

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
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    all_words = []
    for filename in filename_list:
        all_words.append(get_words(filename))
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]

# 构建特征向量
top_words = get_top_words(100)
vector = []
for words in [get_words(f'邮件_files/{i}.txt') for i in range(151)]:
    word_map = list(map(lambda word: words.count(word), top_words))
    vector.append(word_map)
vector = np.array(vector)
labels = np.array([1]*127 + [0]*24)

# 新增数据划分
X_train, X_test, y_train, y_test = train_test_split(
    vector,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# 模型训练与评估
model = MultinomialNB()
model.fit(X_train, y_train)

# 在测试集上生成预测报告
y_pred = model.predict(X_test)
print("\n模型分类评估报告：")
print(classification_report(y_test, y_pred, target_names=['普通邮件', '垃圾邮件']))

def predict(filename):
    """对未知邮件分类"""
    words = get_words(filename)
    current_vector = np.array(tuple(map(lambda word: words.count(word), top_words)))
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'

# 对新邮件的预测保持不变
print('\n新邮件预测结果：')
print('151.txt分类情况:{}'.format(predict('邮件_files/151.txt')))
print('152.txt分类情况:{}'.format(predict('邮件_files/152.txt')))
print('153.txt分类情况:{}'.format(predict('邮件_files/153.txt')))
print('154.txt分类情况:{}'.format(predict('邮件_files/154.txt')))
print('155.txt分类情况:{}'.format(predict('邮件_files/155.txt')))