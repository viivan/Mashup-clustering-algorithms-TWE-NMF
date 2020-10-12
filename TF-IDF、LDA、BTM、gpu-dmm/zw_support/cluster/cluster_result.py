# 训练聚类测试
import sys
sys.path.append('../judge/')
import judge.clustereffect as ce


def printResult(k, result, former):
    pur = ce.purityClusterResult(k, result, former)
    ri = ce.R1ClusterResult(k, result, former)
    # f1 = ce.f1measureClusterResult(k, result, former)
    en = ce.entropyClusterResult(k, result, former)
    pre = ce.precision_cluster(k, result, former)
    recall = ce.recall_cluster(k, result, former)
    f1 = ce.f1measure_cbq(pre, recall)

    #print("纯度:{}, RI:{}, 熵：{}, 准确率：{}，召回率：{}, F1_measure:{}".format(pur, ri, en, pre, recall, f1))
    result_list = [pur, ri, en, pre, recall, f1]
    return result_list