'用于对比实验结果'
import os
import numpy as np
import pandas as pd
from scipy import sparse
import clustereffect as ce
import kmeans as km
import itertools
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
import word_divide as wd;
from sklearn.mixture import GaussianMixture as GMM
import coordinatepainting as cp
from sklearn.cluster import SpectralClustering  as SC
import sklearn.metrics as metrics
from sklearn.decomposition import NMF

DATA_DIR = './dataset/CoEmbedding/'
file = "results_parallel/WT_NMF_K100_iter49.npz"
file1 = "results_parallel/Embeddings_K50_iter19.npz"
data = np.load(os.path.join(DATA_DIR, file))
dir_label = "dataset/C12.csv";
dwmatrix_pt = DATA_DIR+'dw_matrix.csv'
# 文档-主题矩阵
theta = data['U']
# 单词嵌入矩阵
beta = data['B']
# 单词-主题嵌入矩阵
topic = data['V']
n_docs = theta.shape[0]
n_words = topic.shape[0]

global label
def loadModel(save_file):
    # 获取路径信息
    path = os.path.abspath("dataset/" + save_file)

    # 读取文档分布矩阵
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
def dimension_down(data):
    # data就认为是list组好了
    r = np.array(data)
    tsne = TSNE(n_components=2)
    x = tsne.fit_transform(r)
    x_min = np.min(x, 0)  # 获取整个矩阵中最左下的坐标点
    x_max = np.max(x, 0)  # 获取整个矩阵中最右上的坐标点
    x = (x - x_min) / (x_max - x_min)  # 向量归一化
    return x.tolist()
def TCSW():
    data = np.load(os.path.join(DATA_DIR, 'tfidf-data.npy'))
    indices = np.load(os.path.join(DATA_DIR, 'tfidf-indices.npy'))
    indptr = np.load(os.path.join(DATA_DIR, 'tfidf-indptr.npy'))
    matrixD = sparse.csr_matrix((data, indices, indptr), shape=(n_docs, n_words))

    matrixD.data = matrixD.data
    return matrixD
def tfidf(D, normalize=True):

    tf = D.toarray()
    tf[tf>0] = 1
    idf = np.sum(tf, axis=0, keepdims=True)
    idf = np.log(n_docs/idf)
    #归一化
    if normalize:
        D.data = np.log(D.data)+1
        tf = D.toarray()
    return sparse.csr_matrix(tf*idf)

def printResult(k, result, former):
    pur = ce.purityClusterResult(k, result, former)
    ri = ce.R1ClusterResult(k, result, former)
    en = ce.entropyClusterResult(k, result, former)
    pre = ce.precision_cluster(k, result, former)
    recall = ce.recall_cluster(k, result, former)
    f1 = 2*pre*recall/(pre+recall)

    #print("纯度:{}, RI:{}, F1_measure:{}, 熵：{}, 准确率：{}， 召回率：{}".format(pur, ri, f1, en, pre, recall))
    result_list = [pur, ri, f1, en, pre, recall]
    return result_list

#加载标签
def getlabel():
    csv = pd.DataFrame(pd.read_csv(dir_label))
    apiList = csv[["primary_category"]].loc[0:]
    apiData = np.array(apiList).tolist()
    # 存放标签类型
    c_type = set()
    former = []
    for dataItem in apiData:
        c_type.add(dataItem[0].strip())
    c_type = list(c_type)
    print(c_type)
    for dataItem in apiData:
        former.append(c_type.index(dataItem[0].strip()))
    return former

def cluster(X,k):
    #结果取30词平均

    reslut = np.zeros(6)
    for i in range(30):
        # kmeans 算法
        # res = km.kMeansByFeature(k,X)
        # labels = res.labels_
        # 谱聚类
        # assign_labels = "discretize" 离散化 取优值
        # labels = SC(gamma=1e-7, n_clusters=k).fit_predict(X)
        labels = SC(assign_labels = "discretize", gamma=1e-7, n_clusters=k).fit_predict(X)
        # labels = SC( affinity="nearest_neighbors",n_neighbors=10, n_clusters=k).fit_predict(X)
        reslut += np.array(printResult(k, labels, label))
    reslut = reslut / 30;
    print("纯度:{}, RI:{}, F1_measure:{}, 熵：{}, 准确率：{}， 召回率：{}".format(reslut[0], reslut[1], reslut[2],
                                                                   reslut[3], reslut[4], reslut[5]))
def vis(X,title):
    # 参数高维可视化
    x = dimension_down(X)
    cp.printClusterByPointInD2(x,label,title,-1)
    plt.show()

def TNMF(matrixD,K,title = "NMF"):

    model = NMF(n_components=22, init='random', random_state=98765)
    W = model.fit_transform(matrixD)
    H = model.components_
    vis(W,title)
    cluster(W,K)

def GCLM(X,K,title):
    # 高维结果可视化
    vis(X,title)
    cluster(X,K)

if __name__ == '__main__':
    # 获得TF-IDF矩阵
    K = 12
    tp = pd.read_csv(dwmatrix_pt)
    rows, cols = np.array(tp['doc_id']), np.array(tp['word_id'])
    matrixD = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=(n_docs, n_words))
    matrixD = tfidf(matrixD,normalize=True)

    label = getlabel()
    #TCSW方法
    #matrixD = TCSW()
    matrixD = matrixD.toarray()

    # np.savetxt(DATA_DIR+'test.txt', theta)
    # model = "btm_result_12.txt";
    # result = np.array(loadModel(model))
    # print(result);
    # vis(result,"btm");

    #TNMF(matrixD, K)
    GCLM(theta, K,"TWE-NMF")
    # print(theta)
    # CLM + ALS 方法( 速度慢 )
    theta = np.load(os.path.join(DATA_DIR, file1))['U']
    GCLM(theta, K, "CLM")