import os

import cluster.cluster_result as cr
import cluster.kmeans as kmn
import data.data_util as du
import model.TF_IDFAdapter as tfidf
import model.ldaAdapter as ldaa
import model.word2vecAdapter as w2v
import numpy as np


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

# 调用使用gibbs的lda模型
def clusterResult_gibbs(k, topic, model, doc, num=5, sim_num=3, iterator=500):
    # 此时doc为原文本(经过预处理分词)
    # 对文本进行处理，获取tf——idf，使用w2v进行扩容
    # 预计取前5个，扩容为3个
    # num 为 keyword数量
    # sim_num 为 扩容数
    # model = w2v.load_model_binary(r"E:\学校\快乐推荐\word2vec\saveVec")
    print("拓展文档语料")
    doc = tfidf.expend_word(model, doc, num, sim_num)

    # 返回对应的聚类结果
    # 获取lda模型和词袋
    print("创建主题模型")
    word_list, r_model = ldaa.lda_model(doc, k, iterator)

    # 获取文档——主题分布
    doc_topic = r_model.doc_topic_
    writeResult(doc_topic,"12_LDA+wiki.txt");
    # 转为普通list进行聚类
    doc_topic_list = np.array(doc_topic).tolist()
    estimator = kmn.kMeansByFeature(topic, doc_topic_list)
    labels = estimator.labels_

    return list(labels)


if __name__ == "__main__":
    r_k = 12
    r_topic = 12

    file_name = "data12.csv"
    doc = du.getDocAsWordArray(file_name, 5)
    # 获取标签信息

    former = du.getFormerCategory(file_name)
    model = w2v.load_model_binary(r"E:\学校\快乐推荐\word2vec\saveVec")

    result = clusterResult_gibbs(r_k, r_topic, model, doc)
    result_list = cr.printResult(r_topic, result, former)
    print(result_list)

