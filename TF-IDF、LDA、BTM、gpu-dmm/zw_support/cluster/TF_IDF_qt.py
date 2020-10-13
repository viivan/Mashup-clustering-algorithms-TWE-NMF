"""
tf——idf+余弦距离进行计算
KMeans没准要自己实现
"""
import cluster.cluster_result as cr
import data.data_util as du
import model.TF_IDFAdapter as tfidf
import numpy as np


def cal_Cosine_distance(vec0, vec1):
    # 计算余弦相似度,越接近1越相似,参数为两个向量list
    # dot为叉乘
    vec0 = np.array(vec0)
    vec1 = np.array(vec1)
    vector0Mod = np.sqrt(vec0.dot(vec0))
    vector1Mod = np.sqrt(vec1.dot(vec1))
    if vector0Mod != 0 and vector1Mod != 0:
        similarity = abs((vec0.dot(vec1))) / (vector0Mod * vector1Mod)
    else:
        similarity = 0
    return similarity


def qt(doc_vec, tmp, threshold=0.6):
    # 说白了就是设置对应阈值，阈值内的都塞到一个簇中，再将最大的那份给拿出来
    # 返回为最大的那份簇群
    candidate = []
    for i in range(len(doc_vec)):
        i_cluster = []
        for j in range(len(doc_vec)):
            if i == j:
                continue
            if (i in tmp) and (j in tmp):
                ij_sim = cal_Cosine_distance(doc_vec[i], doc_vec[j])
                if ij_sim > threshold:
                    i_cluster.append(j)
        candidate.append(i_cluster)

    # 找到最大的簇返回
    num_s = [len(x) for x in candidate]
    max_num = max(num_s)
    index = num_s.index(max_num)

    return candidate[index]


def eliminate(e, whole_num):
    # 剔除之前簇中元素，whole_num为总的文档数量
    tmp = []
    for i in range(0, whole_num):
        if i in e:
            continue
        else:
            tmp.append(i)
    return tmp


def clusterResult_qt(k, doc):
    # 没整完就直接扔了算了
    threshold = 0.05

    whole_num = len(doc)
    remain = list(range(whole_num))

    # 总之先把tf-idf矩阵求出来
    word_dic, weight = tfidf.cal_tf_idf(doc)
    t = 0
    used = []
    s_num = 0
    cluster_result = []
    out_remain = []
    while t < k - 1:
        candidate = qt(weight, remain, threshold)  # 当前剩余元素中最大簇
        used += candidate
        remain = eliminate(used, whole_num)  # 剔除已用
        out_remain = remain
        s_num += len(candidate)
        print("{}, {}".format(t, len(candidate)))
        cluster_result.append(candidate)
        t += 1
    print("{}, {}".format(t, len(out_remain)))
    cluster_result.append(out_remain)
    print(s_num)

    # 将结果转为可进行评估的方式,未被聚类的设置为-1
    result = [-1] * whole_num
    for i in range(whole_num):
        for x in range(len(cluster_result)):
            if i in cluster_result[x]:
                result[i] = x
                break

    return result


if __name__ == "__main__":
    k = 12
    filename = "data1322.csv"
    # 4 diping 3  8,12 diping 5
    doc = du.getDocAsWordArray(filename,5)

    cluster_result = clusterResult_qt(k, doc)

    former = du.getFormerCategory(filename)
    # print(former)

    # 将数据转为能直接进行准确度判定的形式
    c_r_result = []
    f_r = []
    for i in range(len(cluster_result)):
        if cluster_result[i] != -1:
            c_r_result.append(cluster_result[i])
            f_r.append(former[i])

    # print(len(c_r_result))
    # print(len(f_r))

    r = cr.printResult(k, c_r_result, f_r)

    print(r)



