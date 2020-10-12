"""
利用lda绘制主题词
"""

import data.data_util as du
import data.dimension_reduce as dr
import model.ldaAdapter as ldaa
import numpy as np


def sortSecond(element):
    return element[1]


def getTopicWordAndWordVec(word_list, topic_word, topic_num=12, word_num=5):
    # 为了效果好一点用关键词来搜索
    words = ['price', 'phone', 'google', 'music', 'travel']

    # 参数为对应的主题数和对应的选择的词个数
    # 默认按顺序选择主题
    topic = len(topic_word)
    if topic_num > topic:
        topic_num = topic
    topic_word_index = []

    for i in range(topic_num):
        simple_topic_word = topic_word[i]
        simple_topic_word_index = []

        # 将index和对应概率组合
        simple_tuple = zip(range(len(simple_topic_word)), list(simple_topic_word))
        simple_tuple = list(simple_tuple)
        simple_tuple.sort(key=sortSecond, reverse=True)

        for j in range(word_num):
            simple_topic_word_index.append(simple_tuple[j][0])
        topic_word_index.append(simple_topic_word_index)

    # 先输出下对应词看看
    topic_word_list = []
    for i in range(len(topic_word_index)):
        s_twl = []
        for j in range(len(topic_word_index[i])):
            s_twl.append(word_list[topic_word_index[i][j]])
        topic_word_list.append(s_twl)

    print(topic_word_list)

    # 获取对应词的向量
    word_topic = np.array(topic_word).T
    word_vec = []
    for i in range(len(topic_word_index)):
        s_wv = []
        for j in range(len(topic_word_index[i])):
            s_wv.append(list(word_topic[topic_word_index[i][j]]))
        word_vec.append(s_wv)

    # 输出
    d = []
    whole_d = []
    for wv in word_vec:
        for w in wv:
            whole_d.append(w)
    print("降维开始")
    whole_d = dr.dimension_down(whole_d)

    column = word_num
    row = len(whole_d) / word_num
    print(column, row)
    for i in range(int(row)):
        d.append([0] * column)
    for i in range(int(row)):
        for j in range(int(column)):
            d[i][j] = whole_d[int(i * column + j)]
    print(d)

    # 依据关键词来找
    final_index = []
    for w in words:
        for i in range(len(topic_word_list)):
            if w in topic_word_list[i]:
                if i not in final_index:
                    final_index.append(i)
                    break

    final_word_list = []
    final_word_vec = []
    for i in final_index:
        final_word_list.append(topic_word_list[i])
        final_word_vec.append(d[i])
    print(final_word_list)
    print(final_word_vec)

    return final_word_list, final_word_vec


if __name__ == "__main__":
    csv_filename = "data1322.csv"

    doc = du.getDocAsWordArray(csv_filename)
    r_word_list, model = ldaa.lda_model(doc, 12, 500)
    getTopicWordAndWordVec(r_word_list, list(model.topic_word_))

