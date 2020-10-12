import os

import data.data_util as du
import lda
import lda.datasets
import numpy as np

"""
使用利用gibbs采样实现的lda模型训练
多次迭代并有迭代效果计算
主要是进行对应的数据处理即可
"""


def lda_model(doc, topic, iterator=500):
    # 返回词汇表和训练好的lda模型
    word_set = set()
    print("转化doc_word")
    # 首先创建词汇表
    for d in doc:
        document = doc[d]
        document_word_list = document.split(" ")
        for w in document_word_list:
            word_set.add(w)

    # 创建document矩阵
    N = len(doc)
    V = len(word_set)
    data = []
    word_list = list(word_set)
    for d in doc:
        """
        将存在的词的编号和数量作为tuple存储
        N为文档数，V为词汇数
        之后将其转换为np N*V大小的矩阵
        """
        document = doc[d]
        document_word_list = document.split(" ")
        simple_list = []

        # 每个单词在该文档中包含数
        for i in range(len(word_list)):
            c = document_word_list.count(word_list[i])
            if c > 0:
                simple_list.append((i, document_word_list.count(word_list[i])))
        data.append(tuple(simple_list))

    # 创建矩阵
    dtm = np.zeros((N, V), dtype=np.intc)
    for i, doc in enumerate(data):
        for v, cnt in doc:
            np.testing.assert_equal(dtm[i, v], 0)  # 确认下以免出错
            dtm[i, v] = cnt

    print("训练lda模型")
    model = lda.LDA(n_topics=topic, n_iter=iterator, random_state=1)
    model.fit(dtm)
    return word_list, model


def writeResult(distribution, filename):
    # 太慢了，跑一次要一万年，找个文件写结果
    # distribution就是文档主题分布矩阵

    # 获取路径信息
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = curPath[:curPath.find("zw_support\\") + len("zw_support\\")]
    path = os.path.abspath(rootPath + "result\\" + filename)

    pf = open(path, "w+")
    for dis in distribution:
        str_single = ""
        for d in dis:
            str_single += str(d)+" "
        pf.writelines(str_single[0:len(str_single)-1]+"\n")
    pf.close()


def loadModel(save_file):
    # 获取路径信息
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = curPath[:curPath.find("zw_support\\") + len("zw_support\\")]
    path = os.path.abspath(rootPath + "result\\" + save_file)

    try:
        pf = open(path, "r+")
        result = []  # 存放文档分布
        lines = pf.readlines()
        for line in lines:
            line = line.strip()
            nums = line.split(" ")
            result.append([float(i) for i in nums])
        pf.close()
        return result
    except IOError:
        print("文件无法正常打开")
        return None


if __name__ == "__main__":
    csv_filename = "data10.csv"

    doc = du.getDocAsWordArray(csv_filename)
    r_word_list, model = lda_model(doc, 10)
    doc_topic = list(model.doc_topic_)
    topic_word = list(model.topic_word_)
    writeResult(doc_topic, "10_lda_doc_topic.txt")
    """
    writeResult(doc_topic, "12_lda_doc_topic.txt")
    title = "LDA"
    print("正在降维")
    d2_data = dr.dimension_down(doc_topic)
    cp.paintModelPoint(d2_data, title)
    """
