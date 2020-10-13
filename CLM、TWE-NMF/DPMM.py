import numpy

class DPMM:
    def __init__(self, K0=1, alpha=0.5, beta=0.5,docs=None, V=None, iterNum = 100):
        """
        :param K0:      设定默认主题数K，K会随着算法发生改变;
        :param alpha:   参数α
        :param beta:    参数β
        :param docs:    文档
        :param V:       单词数量
        :param iterNum: Gibbs采样迭代次数
        """
        self.K = K0
        self.V = V
        self.D = len(docs)
        self.alpha = alpha
        self.beta = beta
        self.iterNum = iterNum

        """
        矩阵初始化为0
        """
        self.z = numpy.zeros((self.D),dtype = int)   #每个文档的主题
        self.n_z = numpy.zeros(self.K)               #每个主题下单词的数量
        self.n_zv = numpy.zeros((self.K, V))         #每个单词的在不同主题下的数量
        self.m_z = numpy.zeros(self.K)               #每个主题下文档数量

    # 初始化簇
    def intialize_cluster(self,docs):
        #记录已存在的主题
        count = dict()

        #初始化主题位置,重新赋值避免数组越界
        j = -1;
        for i in range(self.D):
            if self.z[i] not in count:
                j = j + 1;
                count[self.z[i]] = j
            else:
                count[self.z[i]] = count[self.z[i]]


        #print(self.z)
        self.K = len(count)
        #print(self.K)
        for i in range(self.D):
            self.z[i] = count[self.z[i]]

        # K发生改变，矩阵形状重塑
        self.n_z = numpy.zeros(self.K)
        self.n_zv = numpy.zeros((self.K, self.V))
        self.m_z = numpy.zeros(self.K)


        #统计每个主题下单词和文档的数量，
        for i in range(self.D):
            d = docs[i];
            cluster = self.z[i]
            self.m_z[cluster] += 1;
            for wordNo in d:
                wordFre = d[wordNo]
                self.n_zv[cluster][wordNo] += wordFre
                self.n_z[cluster] += wordFre


    # 主题簇选择
    def sampleCluster(self, docs, d):

        doc = docs[d]

        prob = numpy.zeros(self.K+1)
        for k in range(self.K):
            # 统计没有产生新主题的概率
            prob[k] = (self.m_z[k]) / (self.D - 1 + self.alpha);
            valueOfRule2 = 1.0;
            i = 0;
            #对单词统计主题概率
            for wordNo in doc:
                wordFre = doc[wordNo]
                for j in range(wordFre):
                    valueOfRule2 *= (self.n_zv[k][wordNo] + self.beta + j) / (self.n_z[k] + self.V * self.beta + i);
                    i += 1;

            prob[k] = prob[k] * valueOfRule2;

        #计算新主题的概率
        prob[self.K] = (self.alpha) / (self.D - 1 + self.alpha)
        valueOfRule3 = 1.0
        i = 0;

        for w in doc:
            wordFre = doc[w]
            for j in range(wordFre):
                valueOfRule3 *= (self.beta + j) / (self.beta*self.V + i);
                i += 1;

        prob[self.K] = prob[self.K] * valueOfRule3

        #轮盘赌选选择主题
        for k in range(1,self.K+1):
            #print(k)
            prob[k] += prob[k - 1];

        thred = numpy.random.rand() * prob[self.K]
        #print(numpy.random.rand())
        kchoose = 0
        for k in range(self.K+1):
            if thred < prob[k]:
                kchoose = k
                break

        #print(kchoose)
        return kchoose;

    # Gibbs采样
    def gibbsSampling(self,docs):
        for i in range(iterNum):
            #print(i);
            self.intialize_cluster(docs)

            for d in range(self.D):

                doc = docs[d]
                cluster = self.z[i]
                self.m_z[cluster] -= 1;
                for wordNo in doc:
                    wordFre = doc[wordNo]
                    self.n_zv[cluster][wordNo] -= wordFre
                    self.n_z[cluster] -= wordFre

                #轮盘赌选选择主题数
                choose_cluster = self.sampleCluster(docs,d)
                self.z[d] = choose_cluster

                #判断主题数是否增加
                if choose_cluster < self.K:
                    self.m_z[cluster] += 1;
                    for wordNo in doc:
                        wordFre = doc[wordNo]
                        self.n_zv[cluster][wordNo] += wordFre
                        self.n_z[cluster] += wordFre
                #主题数增加，改变矩阵形状
                else:
                    self.intialize_cluster(docs)
        #self.intialize_cluster(docs)

DATA_DIR = './dataset/'
if __name__ == '__main__':
    #统计文本
    docs = {}
    dwid_pt = DATA_DIR + 'CoEmbedding/doc_id.txt'
    wf = open(dwid_pt, 'r')
    i = 0
    K0 = 0
    V = 576
    iterNum = 50
    # a 0.01 b 0.005
    alpha = 0.01
    beta = 0.1
    for l in wf:
        word = {}
        doc = l.strip().split()
        for w in doc:
            w = int(w)
            if w not in word:
                word[w] = 1;
            else:
                word[w] += 1;
        docs[i] = word;
        i += 1;
    model = DPMM(K0, alpha, beta, docs, V, iterNum)

    model.gibbsSampling(docs)
    print(model.K)
    #print(docs)














