"""
对java文件中处理好的gpudmm文档-主题分布文件进行处理
读取文件，调用KMeans方法聚类
"""

import cluster.cluster_result as cr
import data.data_util as du
import model.gpudmmAdapter as gda
import numpy as np
from sklearn.cluster import SpectralClustering  as SP


def clusterResult(k, file_name,former):

    # 获取已经创建好的模型信息
    print("加载文档-主题矩阵")
    result = gda.loadModel(file_name)
    data = np.array(result)

    print("开始kMeans聚类")
    result = np.zeros(6);
    for i in range(30):
        # estimator = kmn.kMeansByFeature(k, data)
        # labels = estimator.labels_
        # SC
        labels = SP(gamma=1e-7, n_clusters=k).fit_predict(data)
        # labels = SP(affinity="nearest_neighbors", n_clusters=k).fit_predict(data)

        result += cr.printResult(k, labels, former)
    return result / 30;


if __name__ == "__main__":
    topic = 12
    filename = "12_gpudmm_doc_topic.txt"

    doc_file = "data1322.csv"
    former = du.getFormerCategory(doc_file)

    res  = clusterResult(topic, filename,former)
    #result = cr.printResult(topic, label, former)
    print(res)