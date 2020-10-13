"""
利用标签进行聚类效果评价
输入为对应的result聚类结果list和former标签知识结果
纯度，RI，熵，F1
"""

# F1 measure 值不对

import math
import numpy as np


def fact(n):
    # 计算阶乘
    if n == 0:
        return 1
    fact_sum = 1
    for i in range(n):
        fact_sum *= (i+1)
    return fact_sum


def C(m, n):
    # 计算组合数
    # m为大数
    if m < n:
        t = m
        m = n
        n = t
    k = m-n
    return fact(m)/(fact(n) * fact(k))


def temple_former(k, result=[], former=[]):
    # 利用先验类型信息获取对应聚类结果的原对应矩阵
    # k为聚类数量,temple存放聚类分组结果

    temple = []
    for i in range(k):
        temple.append([])
    for j in range(len(result)):
        # print(j, result[j])
        # 将聚类结果分组
        temple[int(result[j])].append(j)
    # print(temple)
    # 根据先验，获取每个元素原聚类结果
    former_temple = []
    for i in range(k):
        former_temple.append([])
        for j in temple[i]:
            former_temple[i].append(former[j])
    # print(former_temple)
    return former_temple


def purityClusterResult(k, result=[], former=[]):
    # 获取先验矩阵
    # print(result)
    former_temple = temple_former(k, result, former)

    # 获取每个堆中元素最多的类型作为正确聚类数量
    pur_sum = 0
    for i in range(k):
        num_max = 0
        for j in set(former_temple[i]):
            c = former_temple[i].count(j)
            if c > num_max:
                num_max = c
        pur_sum += num_max
    return pur_sum/len(result)


def entropyClusterResult(k1, result=[], former=[]):
    former_set = set(former)  # 获取类别
    former_temple = temple_former(k1, result, former)  # 获取先验矩阵

    # 计算每个聚类的ei
    e = 0.0
    for i in range(k1):
        # print("turn:", i)
        ei_simple = 0
        m = len(former_temple[i])

        # print("m:", m)

        # 每一类
        for j in former_set:
            p = former_temple[i].count(j) / m
            # print("j:{}p:{}".format(j, p))
            if p == 0:
                continue
            ei_simple += p * math.log(p, 2)
        ei = (-ei_simple) * m / len(result)
        e += ei

    return e


def precisionResult(k, result=[], former=[]):
    former_set = set(former)  # 获取类别
    former_temple = temple_former(k, result, former)  # 获取先验矩阵
    # 计算TP，FP
    tp = 0
    tp_fp = 0
    for i in range(k):
        size = len(former_temple[i])
        if size < 2:
            continue
        tp_fp += C(size, 2)
    for i in range(k):
        for j in former_set:
            # print(former_temple[i].count(j))
            if former_temple[i].count(j) >= 2:
                tp += C(former_temple[i].count(j), 2)
    p = tp / tp_fp
    return p


def R1ClusterResult(k, result=[], former=[]):
    former_set = set(former)  # 获取类别
    former_temple = temple_former(k, result, former)  # 获取先验矩阵
    # 计算TP，FP
    tp = 0
    tp_fp = 0
    for i in range(k):
        size = len(former_temple[i])
        if size < 2:
            continue
        tp_fp += C(size, 2)
    # 计算tp,对于每个聚类簇进行处理
    for i in range(k):
        for j in former_set:
            # print(former_temple[i].count(j))
            if former_temple[i].count(j) >= 2:
                tp += C(former_temple[i].count(j), 2)
    # print("tp_fp:{},tp:{}".format(tp_fp, tp))

    # 计算TN，FN
    fn = 0
    tn_fn = 0
    for i in range(k):
        for j in range(i + 1, k):
            tn_fn += len(former_temple[i]) * len(former_temple[j])
    # print("tn_fn:{}".format(tn_fn))
    # 计算fn，对每个簇进行处理
    for i in range(k):
        for j in former_set:
            now_sum = former_temple[i].count(j);
            if now_sum == 0:
                continue
            last_sum = 0  # 存放剩余簇中所有该类数据的总和
            for g in range(i+1, k):
                last_sum += former_temple[g].count(j)
            if last_sum == 0:
                continue
            fn += now_sum * last_sum
    # print("fn:{}".format(fn))

    tn = tn_fn - fn
    r1 = (tp + tn) / (tn_fn + tp_fp)
    return r1


def f1measureClusterResult(k, result=[], former=[]):
    former_set = set(former)  # 获取类别
    former_temple = temple_former(k, result, former)  # 获取先验矩阵
    # 计算TP，FP
    tp = 0
    tp_fp = 0
    for i in range(k):
        size = len(former_temple[i])
        if size < 2:
            continue
        tp_fp += C(size, 2)
    # 计算tp,对于每个聚类簇进行处理
    for i in range(k):
        for j in former_set:
            # print(former_temple[i].count(j))
            if former_temple[i].count(j) >= 2:
                tp += C(former_temple[i].count(j), 2)
    # print("tp_fp:{},tp:{}".format(tp_fp, tp))

    # 计算TN，FN
    fn = 0
    tn_fn = 0
    for i in range(k):
        for j in range(i + 1, k):
            tn_fn += len(former_temple[i]) * len(former_temple[j])
    # print("tn_fn:{}".format(tn_fn))
    # 计算fn，对每个簇进行处理
    for i in range(k):
        for j in former_set:
            now_sum = former_temple[i].count(j);
            if now_sum == 0:
                continue
            last_sum = 0  # 存放剩余簇中所有该类数据的总和
            for g in range(i+1, k):
                last_sum += former_temple[g].count(j)
            if last_sum == 0:
                continue
            fn += now_sum * last_sum
    # print("fn:{}".format(fn))

    p = tp / tp_fp
    r = tp / (tp + fn)
    under = p + r
    if under != 0:
        f = 2 * p * r / under
        # print("under{} p{} r{}".format(under, p, r))
        return f
    else:
        return 0


# 按照cbq论文中准确度进行计算，将分布和对应的预先分类进行映射
# 两个聚类结果簇数量相同，对应
def precision_cluster(k, result=[], former=[]):
    """
    先将result id放入对应聚类
    将former也相同操作
    在一个list中放入former簇对应的result簇编号
    """
    temple = []
    for i in range(k):
        temple.append([])
    for j in range(len(result)):
        # print(j, result[j])
        # 将聚类结果分组
        temple[int(result[j])].append(j)
    # for i in temple:
    #     print(len(i))
    f_temple = []
    for i in range(k):
        f_temple.append([])
    for j in range(len(former)):
        # print(j, result[j])
        # 将聚类结果分组
        f_temple[int(former[j])].append(j)

    # 计算result各簇中对应的 原簇元素数量/簇元素总数
    sm_m = []
    for i in range(k):
        s_max = []
        for j in range(k):  # 对每个分类计算
            s = 0
            for x in temple[i]:
                if x in f_temple[j]:
                    s += 1
            s_max.append(s / len(temple[i]))
        sm_m.append(s_max)
    # for  i in sm_m:
    #       print(i)
    # 将最大的sm_m提取，获取对应的簇对应
    corresponding = [0] * k
    used = [0] * k  # 已被占用簇
    for i in range(k):
        s_max = 0
        index_max = 0
        for j in range(k):
            if used[j] != 1:
                if sm_m[i][j] >= s_max:
                    s_max = sm_m[i][j]
                    index_max = j
        used[index_max] = 1
        corresponding[i] = index_max
    #print(corresponding)
    p = 0
    for i in range(k):
        p += sm_m[i][corresponding[i]]
    p = p / k
    return p


def recall_cluster(k, result=[], former=[]):
    # 和准确率类似，最终使用smi进行相除处理
    temple = []
    for i in range(k):
        temple.append([])
    for j in range(len(result)):
        # print(j, result[j])
        # 将聚类结果分组
        temple[int(result[j])].append(j)

    f_temple = []
    for i in range(k):
        f_temple.append([])
    for j in range(len(former)):
        # print(j, result[j])
        # 将聚类结果分组
        f_temple[int(former[j])].append(j)

    # 计算result各簇中对应的 原簇元素数量/簇元素总数
    sm_m = []
    for i in range(k):
        s_max = []
        for j in range(k):  # 对每个分类计算
            s = 0
            for x in temple[i]:
                if x in f_temple[j]:
                    s += 1
            s_max.append(s / len(temple[i]))
        sm_m.append(s_max)

    # 将最大的sm_m提取，获取对应的簇对应
    corresponding = [0] * k
    used = [0] * k  # 已被占用簇
    for i in range(k):
        s_max = 0
        index_max = 0
        for j in range(k):
            if used[j] != 1:
                if sm_m[i][j] >= s_max:
                    s_max = sm_m[i][j]
                    index_max = j
        used[index_max] = 1
        corresponding[i] = index_max
    # 计算回归率
    # print(temple)
    # print(f_temple)
    # print(corresponding)
    r = 0
    for i in range(k):
        right = sm_m[i][corresponding[i]] * len(temple[i]) / len(f_temple[corresponding[i]])
        if(right > 1):
            print(sm_m[i][corresponding[i]] * len(temple[i]),len(f_temple[corresponding[i]]))
        r += right
    r = r / k
    return r


if __name__ == "__main__":
    # former_t = [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 2, 0, 0, 2, 2, 2]
    # result_t = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    result_t = [0, 0, 1, 1, 2, 2]
    former_t = [0, 0, 0, 1, 2, 2]
    #print(np.eye(20, dtype=float))
    #precision = precision_cluster(3, result_t, former_t)
    #print("准确率:", precision)
    #recall = recall_cluster(3, result_t, former_t)
    #print("回归率:", recall)