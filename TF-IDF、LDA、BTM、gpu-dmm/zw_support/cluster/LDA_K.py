import model.ldaAdapter as ldaa
import numpy as np
from sklearn.cluster import SpectralClustering  as SC
from cluster import kmeans as kmn
import data.data_util as du
import cluster.cluster_result as cr

"""
把几个接口封装下
方便用gibbs采样的lda
"""


def lda_kmn_result(k, topic, doc, former ,iterator=1000):

    # 返回对应的聚类结果
    # 获取lda模型和词袋
    print("创建主题模型")
    word_list, r_model = ldaa.lda_model(doc, topic, iterator)

    # 获取文档——主题分布
    doc_topic = r_model.doc_topic_
    # 转为普通list进行聚类
    doc_topic_list = np.array(doc_topic).tolist()

    result = np.zeros(6);
    for i in range(30):
        estimator = kmn.kMeansByFeature(topic, doc_topic_list)
        labels = estimator.labels_
        # assign_labels="discretize",
        #labels = SC(assign_labels="discretize",gamma=1e-7, n_clusters=k).fit_predict(doc_topic)
        result += cr.printResult(k, labels, former)
    return result/30;


if __name__ == "__main__":
    r_k = 4
    r_topic = 4
    file_name = "data4.csv"
    doc = du.getDocAsWordArray(file_name, 5)
    # 获取标签信息
    former = du.getFormerCategory(file_name)

    result = lda_kmn_result(r_k, r_topic, doc, former, 1000)
    # print("纯度:{}, RI:{}, 熵：{}, 准确率：{}，召回率：{}, F1_measure:{}".format(pur, ri, en, pre, recall, f1))
    print(result)