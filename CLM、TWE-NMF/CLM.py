import sys
import numpy as np
import pandas as pd
from scipy import sparse
import os
import CoEmbedding

DATA_DIR = './dataset/CoEmbedding/'
dwmatrix_pt = DATA_DIR+'dw_matrix.csv'
vocab_pt = DATA_DIR+'vocab.txt'
#10
# n_docs = 883
# n_words = 576
#8
# n_docs = 734
# n_words = 576
#4
# n_docs = 392
# n_words = 454
#12
n_docs = 1340
n_words = 744
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
#改进方案
def tfidf1():

    data = np.load(os.path.join(DATA_DIR, 'tfidf-data.npy'))
    indices = np.load(os.path.join(DATA_DIR, 'tfidf-indices.npy'))
    indptr = np.load(os.path.join(DATA_DIR, 'tfidf-indptr.npy'))
    matrixD = sparse.csr_matrix((data, indices, indptr), shape=(n_docs, n_words))

    # 归一化
    #maxn = data.max()+1;
    #print(maxn)
    matrixD.data = matrixD.data
    return matrixD

# load matrix D
tp = pd.read_csv(dwmatrix_pt)
rows, cols = np.array(tp['doc_id']), np.array(tp['word_id'])
matrixD = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=(n_docs, n_words))
matrixD = tfidf(matrixD, normalize=True)
# print('tf-idf:',matrixD.shape)
# matrixD = tfidf1()
# NFM计算
# load matrix W
data = np.load(os.path.join(DATA_DIR, 'coordinate_co_binary_data.npy'))
indices = np.load(os.path.join(DATA_DIR, 'coordinate_co_binary_indices.npy'))
indptr = np.load(os.path.join(DATA_DIR, 'coordinate_co_binary_indptr.npy'))
matrixW = sparse.csr_matrix((data, indices, indptr), shape=(n_words, n_words))
# see the sparseness
print(matrixD.shape, matrixW.shape)
print(float(matrixD.nnz) / np.prod(matrixD.shape))
print(float(matrixW.nnz) / np.prod(matrixW.shape))


def get_row(Y, i):
    # 每行的开始位置和结束位置
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    # 获取对应的值和坐标
    return lo, hi, Y.data[lo:hi], Y.indices[lo:hi]


count = np.asarray(matrixW.sum(axis=1)).ravel()
n_pairs = matrixW.data.sum()

# 计算 SPPMI 矩阵
M = matrixW.copy()
for i in range(n_words):
    lo, hi, d, idx = get_row(M, i)
    M.data[lo:hi] = np.log(d * n_pairs / (count[i] * count[idx]))
    # M.data[lo:hi] = (n_pairs*d)/(count[idx]*n_words)

print(max(M.data))
print(M[0, 0])

# 将负值置0
M.data[M.data < 0] = 0
M.eliminate_zeros()
print(float(M.nnz) / np.prod(M.shape))
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
print(np.sum(np.absolute(M_ns)) / np.prod(M_ns.shape))
# print(M_ns)
# print(M_ns.toarray())

# 参数设定
n_embeddings = 50
max_iter = 20
n_jobs = 8
c0 = 1
c1 = 1
K = 22
lam_sparse_d = 1e-2
lam_sparse = 1e-7
lam_d = 0.5
lam_w = 1
lam_t = 50
save_dir = os.path.join(DATA_DIR, 'results_parallel')
wukong = CoEmbedding.CoEmbedding(n_embeddings=n_embeddings, K=K, max_iter=max_iter, batch_size=1000, init_std=0.01,
                                 n_jobs=n_jobs,
                                 random_state=98765, save_params=True, save_dir=save_dir, verbose=False,
                                 lam_sparse_d=lam_sparse_d, lam_sparse=lam_sparse, lam_d=lam_d, lam_w=lam_w,
                                 lam_t=lam_t, c0=c0, c1=c1)
wukong.fit(matrixD, M_ns, vocab_pt)



print(wukong.topic.shape)
topicfile = DATA_DIR + 'ourtwords.txt'
topicembeddingfile = DATA_DIR + 'ourtembeddings.txt'
np.savetxt(topicembeddingfile, wukong.alpha)
wukong.show_save_topics(10, topicfile)

#print(wukong.topic_similarity())




