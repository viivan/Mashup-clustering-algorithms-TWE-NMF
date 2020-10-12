import os
import sys
import time
import sklearn
import numpy as np
import pandas as pd
# numpy.linalg模块包含线性代数的函数。使用这个模块，我们可以计算逆矩阵、求特征值、解线性方程组以及求解行列式等。
from numpy import linalg as LA, argpartition

from scipy import sparse

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin


class WT_NMF(BaseEstimator, TransformerMixin):
    def __init__(self, n_embeddings=100, K=10, max_iter=10, batch_size=1000,
                 init_std=0.01, dtype='float64', n_jobs=8, random_state=None,
                 save_params=False, save_dir='.', verbose=False, **kwargs):

        self.n_embeddings = n_embeddings
        self.n_topics = K
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.init_std = init_std
        self.dtype = dtype
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.save_params = save_params
        self.save_dir = save_dir
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.set_state(self.random_state)

        self._parse_kwargs(**kwargs)

    # 参数设定
    def _parse_kwargs(self, **kwargs):


        self.lam_sparse_d = float(kwargs.get('lam_sparse_d', 1e-5))
        self.lam_sparse = float(kwargs.get('lam_sparse', 1e-5))
        # 全局信息系数
        self.lam_d = float(kwargs.get('lam_d', 1e0))
        # 局部信息系数
        self.lam_w = float(kwargs.get('lam_w', 1e-2))

        self.lam_t = float(kwargs.get('lam_t', 1e-2))
        self.c0 = float(kwargs.get('c0', 0.01))
        self.c1 = float(kwargs.get('c1', 1.0))
        assert self.c0 <= self.c1

    # 随机初始化矩阵值
    def _init_params(self, n_docs, n_words):
        ''' 初始化矩阵'''
        # 初始化θ 文档-主题矩阵 NxK
        self.theta = self.init_std * np.random.randn(n_docs, self.n_topics).astype(self.dtype) + 1
        # 初始化T 词-主题矩阵 VxK
        self.topic = self.init_std * np.random.randn(n_words, self.n_topics).astype(self.dtype) + 1
        # 初始化W 词向量矩阵 MXV
        self.beta = self.init_std * np.random.randn(n_words, self.n_embeddings).astype(self.dtype)+1
        # 初始化G 文本嵌入矩阵 NXV
        self.gamma = self.init_std * np.random.randn(n_docs, self.n_embeddings).astype(self.dtype)+1
        # 初始化A 主题嵌入矩阵 KXV
        self.alpha = self.init_std * np.random.randn(self.n_topics, self.n_embeddings).astype(self.dtype)+1
        # 初始化S 对称因子矩阵 VXV
        self.sem = self.init_std * np.random.rand(self.n_embeddings,self.n_embeddings).astype(self.dtype)+1

        assert np.all(self.theta > 0)
        assert np.all(self.topic > 0)

    def fit(self, X, M, voca_pt):
        '''
        X : X为 文档-词的加权稀疏矩阵

        M : 词共现的SPPMI矩阵

        voca_pt : 词汇库
        '''
        self._read_vocab(voca_pt)
        n_docs, n_words = X.shape
        assert M.shape == (n_words, n_words)

        self._init_params(n_docs, n_words)
        # 转成矩阵形式
        X = X.toarray()
        M = M.toarray()
        self._update(X, M)
        return self

    # 更新参数
    def _update(self, X, M):
        '''更新参数'''
        XT = X.T
        for i in range(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)
            self._update_factors(X, XT, M)
            if self.save_params:
                self._save_params(i)

    # 更新相应的参数
    def _update_factors(self, X, XT, M):
        # 更新文档-主题
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating theta...')
        self.theta = update_theta(self.topic, self.theta,self.gamma,self.alpha,X)
        # 更新词-主题
        if self.verbose:
            print('\r\tUpdating T: time=%.2f' % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating topics...')
        self.topic = update_topic(self.topic, self.theta,self.alpha,self.beta,XT,self.lam_d,self.lam_w,self.lam_t)
        self.topic = _normalize_topic(self.topic)
        if(self.lam_t == 0 and self.lam_w == 0) :
            return;
        # 更新词嵌入
        if self.verbose:
            print('\r\tUpdating word embeddings: time=%.2f' % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating word embeddings...')
        self.beta = update_beta(self.topic, self.alpha,self.beta,self.sem,M,self.lam_d,self.lam_w,self.lam_t)
        #更新缩放因子
        if self.verbose:
            print('\r\tUpdating S: time=%.2f' % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating topics...')
        self.sem = update_sem(self.beta, self.sem, M)
        # 更新主题嵌入
        if self.verbose:
            #print('\r\tUpdating topic embeddings: time=%.2f' % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating topic embeddings...')
        self.alpha = update_alpha(self.topic.T, self.theta.T, self.gamma, self.beta,self.alpha)
        if self.verbose:
            print('\r\tUpdating topic embeddings: time=%.2f' % (time.time() - start_t))

        # # 更新文档嵌入
        # if self.verbose:
        #     print('\r\tUpdating topics: time=%.2f' % (time.time() - start_t))
        #     start_t = _writeline_and_time('\tUpdating context embeddings...')
        # self.gamma = update_gamma(self.theta, self.gamma, self.alpha)

    # 保存结果
    def _save_params(self, iter):
        '''Save the parameters'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if iter == self.max_iter-1:
            filename = 'WT_NMF_K%d_iter%d.npz' % (self.n_embeddings, iter)
            np.savez(os.path.join(self.save_dir, filename), U=self.theta, V=self.topic, G=self.gamma, B=self.beta,
                 A=self.alpha)

    # 计算词相似度
    def most_similar(self, dword, topn):
        wid = self.w2id[dword]
        unibeta = self.beta.copy().T
        normss = np.linalg.norm(unibeta, axis=1, keepdims=True)
        unibeta = unibeta / normss
        wvec = unibeta[wid]
        distances = np.inner(-wvec, unibeta)
        most_extreme = np.argpartition(distances, topn)[:topn]
        # print(np.sort(distances.take(most_extreme)))
        return [self.id2w[t] for t in
                most_extreme.take(np.argsort(distances.take(most_extreme)))]  # resort topn into order

    def show_save_topics(self, topn, filename):
        fout = open(filename, 'w')
        topicword = [[] for i in range(self.n_topics)]
        tempT = - self.topic.copy().T
        for i in range(self.n_topics):
            most_extreme = np.argpartition(tempT[i], topn)[:topn]
            topicword[i] = [self.id2w[t] for t in most_extreme.take(np.argsort(tempT[i].take(most_extreme)))]
            fout.writelines('Topic %dth:\n' % i)
            for word in topicword[i]:
                fout.writelines('\t%s\n' % word)
        fout.close()
        return topicword  # resort topn into order

    def show_topics_embeddings(self, topn):
        topicword = [[] for i in range(self.n_topics)]
        unigamma = self.beta.copy().T
        # normss = np.linalg.norm(unigamma, axis = 1, keepdims = True)
        # unigamma = unigamma/normss
        unialpha = self.alpha.copy()
        # normsss = np.linalg.norm(unialpha, axis = 1, keepdims = True)
        # unialpha = unialpha/normsss
        for i in range(self.n_topics):
            tvec = unialpha[i]
            distances = np.inner(-tvec, unigamma)
            most_extreme = np.argpartition(distances, topn)[:topn]
            topicword[i] = [self.id2w[t] for t in most_extreme.take(np.argsort(distances.take(most_extreme)))]
        return topicword  # resort topn into order

    def topic_similarity(self, topn=10):
        topicword = [[] for i in range(self.n_topics)]
        unialpha = self.alpha.copy()
        normsss = np.linalg.norm(unialpha, axis=1, keepdims=True)
        unialpha = unialpha / normsss
        for i in range(self.n_topics):
            tvec = unialpha[i]
            distances = np.inner(-tvec, unialpha)
            # print('topic',distances)
            topicword[i] = np.argsort(distances)
        return topicword

    def _read_vocab(self, voca_pt):
        self.w2id = {}
        self.id2w = {}
        for l in open(voca_pt, encoding="UTF-8"):
            ws = l.strip().split()
            self.w2id[ws[1]] = int(ws[0])
            self.id2w[int(ws[0])] = ws[1]


# Utility functions #
def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


def get_row(Y, i):
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return lo, hi,Y.data[lo:hi], Y.indices[lo:hi]


# 更新文档-主题信息
'''由公式(DT)/(θT^tT)Xθ  X:表示阿达马乘积即对应位置的乘积'''
def update_theta(topic,theta, gamma, alpha, X):
    '''更新主题因子'''

    num = np.dot(X, topic)
    denom = np.dot(theta, topic.T).dot(topic)

    res = np.divide(num, denom)
    res = np.multiply(res, theta)
    # print(res)
    res[res < 0] = 0
    return res

# 更新主题-词矩阵
'''由公式(WA^t+D^tT)/(Tθ^tθ+T)XT  X:表示阿达马乘积即对应位置的乘积'''
def update_topic(topic,theta, alpha, beta, XT,lam_d,lam_w,lam_t):
    '''跟新词-主题矩阵'''
    num = lam_t*np.dot(beta, alpha.T) + lam_d*np.dot(XT, theta)
    denom = lam_d *np.dot(topic, theta.T).dot(theta) + lam_t*topic
    res = np.divide(num, denom)
    res = np.multiply(res, topic)
    res[res < 0] = 0
    return res

# 更新词嵌入矩阵
'''由公式(TA+2MWS)/(2WSW^tWS+WA^tA)XW  X:表示阿达马乘积即对应位置的乘积'''
def update_beta(topic,alpha,beta,sem,M,lam_d,lam_w,lam_t):
    '''更新词嵌入矩阵'''
    WS = lam_t*np.dot(beta, sem)
    num = np.dot(topic, alpha) + lam_w*2*np.dot(M, WS)
    denom = 2*lam_w*WS.dot(np.dot(beta.T, WS)) + lam_t*np.dot(beta, alpha.T).dot(alpha)
    res = np.divide(num, denom)
    res = np.multiply(res, beta)
    # print(res)
    return res

#更新 额外因子
'''由公式(W^tMW)/(W^tWSW^tW)XS  X:表示阿达马乘积即对应位置的乘积'''
def update_sem(beta, sem, M):
    '''更新词-主题矩阵'''
    W = np.dot(beta.T, beta)
    num = np.dot(beta.T, M).dot(beta)
    denom = np.dot(W, sem).dot(W)
    res = np.divide(num, denom)
    res = np.multiply(sem, res)

    return res

# 更新主题嵌入矩阵
'''由公式(T^tW+θG)/(AW^tW+AG^tG)XA  X:表示阿达马乘积即对应位置的乘积'''
def update_alpha(topicT,thetaT,gamma ,beta, alpha):
    '''更新主题嵌入'''
    num = np.dot(topicT, beta)+np.dot(thetaT, gamma)*0
    denom = np.dot(alpha, beta.T).dot(beta) + np.dot(alpha, gamma.T).dot(gamma)*0
    res = np.divide(num, denom)
    res = np.multiply(res, alpha)

    return res

# 更新文本嵌入矩阵
# '''由公式(DA)/(GA^tA)XA  X:表示阿达马乘积即对应位置的乘积'''
# def update_gamma(theta,gamma,alpha,lam_d = 0.5,lam_w = 1,lam_t = 50):
#     '''更新文本嵌入矩阵'''
#     num = np.dot(theta, alpha)
#     denom = np.dot(gamma, alpha.T).dot(alpha)
#
#     res = np.divide(num, denom)
#     res = np.multiply(res, gamma)
#     return res

def _normalize_topic(topic):
    norms = np.sum(topic, axis=0)
    normtopic = topic / norms
    return normtopic

def tfidf(D, normalize=True):

    tf = D.toarray()
    #print(len(tf[0]))
    #print(D)
    #print(tf)
    tf[tf>0] = 1
    #print(tf)
    idf = np.sum(tf, axis=0, keepdims=True)
    idf = np.log(n_docs/idf)
    #print(idf)
    #归一化
    if normalize:
        D.data = np.log(D.data)+1
        tf = D.toarray()
    return sparse.csr_matrix(tf*idf)
def TCSW():

    data = np.load(os.path.join(DATA_DIR, 'tfidf-data.npy'))
    indices = np.load(os.path.join(DATA_DIR, 'tfidf-indices.npy'))
    indptr = np.load(os.path.join(DATA_DIR, 'tfidf-indptr.npy'))
    matrixD = sparse.csr_matrix((data, indices, indptr), shape=(n_docs, n_words))

    # 归一化
    #maxn = data.max()+1;
    #print(maxn)
    matrixD.data = matrixD.data
    return matrixD
if __name__ == "__main__":
    # 测试
    DATA_DIR = './dataset/CoEmbedding/'
    dwmatrix_pt = DATA_DIR + 'dw_matrix.csv'
    vocab_pt = DATA_DIR + 'vocab.txt'
    # 10
    # n_docs = 883
    # n_words = 576
    # 12
    n_docs = 1340
    n_words = 744
    # 8
    # n_docs = 734
    # n_words = 576
    # 4
    # n_docs = 392
    # n_words = 454
    # 参数设定
    n_embeddings = 100
    max_iter = 50
    n_jobs = 9
    c0 = 1
    c1 = 1
    # 4-5, 8-9,10-16, 12-22;
    K = 16
    lam_sparse_d = 1e-2
    lam_sparse = 1e-7
    lam_d = 0.05
    lam_w = 0.05
    lam_t = 50
    save_dir = os.path.join(DATA_DIR, 'results_parallel')

    model = WT_NMF(n_embeddings=n_embeddings, K=K, max_iter=max_iter, batch_size=1000, init_std=0.1,
                                 n_jobs=n_jobs,
                                 random_state=98765, save_params=True, save_dir=save_dir, verbose=True,
                                 lam_sparse_d=lam_sparse_d, lam_sparse=lam_sparse, lam_d=lam_d, lam_w=lam_w,
                                 lam_t=lam_t, c0=c0, c1=c1)

    data = np.load(os.path.join(DATA_DIR, 'coordinate_co_binary_data.npy'))
    indices = np.load(os.path.join(DATA_DIR, 'coordinate_co_binary_indices.npy'))
    indptr = np.load(os.path.join(DATA_DIR, 'coordinate_co_binary_indptr.npy'))
    matrixW = sparse.csr_matrix((data, indices, indptr), shape=(n_words, n_words))

    tp = pd.read_csv(dwmatrix_pt)
    rows, cols = np.array(tp['doc_id']), np.array(tp['word_id'])
    matrixD = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=(n_docs, n_words))
    matrixD = tfidf(matrixD, normalize=True)
    #是否采用TCSW方法
    matrixD = TCSW()
    count = np.asarray(matrixW.sum(axis=1)).ravel()
    n_pairs = matrixW.data.sum()

    # 计算 SPPMI 矩阵
    M = matrixW.copy()
    for i in range(n_words):
        lo, hi, d, idx = get_row(M, i)
        M.data[lo:hi] = np.log(d * n_pairs / (count[i] * count[idx]))
        # M.data[lo:hi] = (n_pairs*d)/(count[idx]*n_words)
    # 将负值置0
    M.data[M.data < 0] = 0
    M.eliminate_zeros()
    # Now $M$ is the PPMI matrix. Depending on the number of negative examples $k$, we can obtain the shifted PPMI matrix as $\max(M_{wc} - \log k, 0)$

    # 非负采样偏差值
    k_ns = 1
    M_ns = M.copy()
    if k_ns > 1:
        offset = np.log(k_ns)
    else:
        offset = 0.

    M_ns.data -= offset
    M_ns.data[M_ns.data < 0] = 0
    M_ns.eliminate_zeros()
    model.fit(matrixD, M_ns, vocab_pt)

    topicfile = DATA_DIR + 'ourtwords_WT.txt'
    topicembeddingfile = DATA_DIR + 'ourtembeddings_WT.txt'
    np.savetxt(topicembeddingfile,model.alpha)
    model.show_save_topics(10, topicfile)

