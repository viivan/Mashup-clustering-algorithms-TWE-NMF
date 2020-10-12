import cluster.cluster_result as cr
import data.data_util as du
import model.btmAdapter as btm
import numpy as np
from sklearn.cluster import SpectralClustering  as SC

"""
利用btm进行模型训练，获取分布信息
利用kMeans进行聚类处理
后续可能会进行相似度预处理
"""


def clusterResult(k, file_name,former):

    # 获取已经创建好的模型信息
    print("加载文档-主题矩阵")
    result = btm.loadModel(file_name)
    data = np.array(result)

    print("开始kMeans聚类")
    result = np.zeros(6);
    for i in range(30):
        #estimator = kmn.kMeansByFeature(k, data)
        #labels = estimator.labels_
        labels = SC(assign_labels="discretize", gamma=1e-7, n_clusters=k).fit_predict(data)
        result += cr.printResult(k, labels, former)
    return result / 30;

    print("聚类完成")

    #return list(estimator.labels_)


if __name__ == "__main__":

    file_name = "data10.csv"

    former = du.getFormerCategory(file_name)
    t_result = clusterResult(10, "btm_result_10.txt",former)

    print(t_result)
    #print(c_result)