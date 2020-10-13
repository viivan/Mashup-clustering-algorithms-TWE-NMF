import cluster.cluster_result as cr
import data.data_util as du
import model.TF_IDFAdapter as tfidf
import numpy as np
from cluster import kmeans as kmn

if __name__ == "__main__":
    k = 4
    filename = "data4.csv"
    # 4 diping 3  8,12 diping 5
    doc = du.getDocAsWordArray(filename,3)
    former = du.getFormerCategory(filename)

    word_dic, weight = tfidf.cal_tf_idf(doc)
    result = np.zeros(6);
    for i in range(30):
        estimator = kmn.kMeansByFeature(k, weight)
        labels = estimator.labels_
        result += cr.printResult(k, labels, former)
    result = result / 30;

    print(result)