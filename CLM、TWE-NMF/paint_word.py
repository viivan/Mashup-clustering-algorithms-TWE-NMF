"""
利用lda绘制主题词
"""
import os
import numpy as np
from sklearn.manifold import TSNE
import coordinatepainting as cp

DATA_DIR = 'C:/Users/赵伟/Desktop/相关代码/TWE-NMF/dataset/CoEmbedding/'
file = "results_parallel/WT_NMF_K100_iter49.npz"
data = np.load(os.path.join(DATA_DIR, file))

w2id = {}
id2w = {}
def sortSecond(element):
    return element[1]

def dimension_down(data):
    # data就认为是list组好了
    r = np.array(data)
    tsne = TSNE(n_components=2)
    x = tsne.fit_transform(r)
    x_min = np.min(x, 0)  # 获取整个矩阵中最左下的坐标点
    x_max = np.max(x, 0)  # 获取整个矩阵中最右上的坐标点
    x = (x - x_min) / (x_max - x_min)  # 向量归一化
    return x.tolist()

def read_vocab(voca_pt):

    for l in open(voca_pt, encoding="UTF-8"):
        ws = l.strip().split()
        w2id[ws[1]] = int(ws[0])
        id2w[int(ws[0])] = ws[1]

def cal_pos(word_vec,word_num):
    d = []
    whole_d = []
    for wv in word_vec:
        for w in wv:
            whole_d.append(w)
    print("降维开始")
    whole_d = dimension_down(whole_d)

    column = word_num
    row = len(whole_d) / word_num
    # print(column, row)
    for i in range(int(row)):
        d.append([0] * column)
    for i in range(int(row)):
        for j in range(int(column)):
            d[i][j] = whole_d[int(i * column + j)]
    return d;
def get_topicwords(tempT, topn, topic_nums,topic_word):

    topicword = [[] for i in range(topic_nums)]
    topicvec = [[] for i in range(topic_nums)]
    for i in range(topic_nums):
        most_extreme = np.argpartition(tempT[i], topn)[:topn]
        #print(most_extreme)
        topicword[i] = [id2w[t] for t in most_extreme.take(np.argsort(tempT[i].take(most_extreme)))]
        for word in topicword[i]:
            #print(w2id[word])
            topicvec[i].append(topic_word[w2id[word]])
    return topicword,topicvec


if __name__ == "__main__":

    # 获得单词信息
    voca_pt = DATA_DIR + 'vocab.txt'
    read_vocab(voca_pt)
    # print(word_list);
    #print(id2w)
    #doc = du.getDocAsWordArray(csv_filename)
    #r_word_list, model = ldaa.lda_model(doc, 12, 500)
    topic_word = data['V']
    #print(topic_word[88])
    #print(topic_word.shape)
    topicword, topicvec = get_topicwords(-topic_word.T,10,17,topic_word);
    dvector = cal_pos(topicvec,5);
    print(dvector)
    cp.printTopicWordByPointInD2(dvector,topicword,"")

