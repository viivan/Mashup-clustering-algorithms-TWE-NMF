'''
Remove words occur less than min_count times throughout the corpus
'''
import sys
import numpy as np
import pandas as pd
import word_divide as wd
from scipy import sparse
import itertools
import os



w2cnt = {}
w2id = {}
DATA_DIR = './dataset/'
# 单词计数
def wordCount(pt):
    print('counting...')
    for l in pt:
        ws = l.strip().split()
        for w in ws:
            w2cnt[w] = w2cnt.get(w,0) + 1
# 去除低频词
def indexFile(pt, out_pt, min_count = 10):
    #print('index file: ', pt)
    wf = open(out_pt, 'w',encoding="UTF-8")
    for l in pt:
        ws = l.strip().split()
        for w in ws:
            if w2cnt[w] >= min_count:
                wf.writelines(w+" ")
            #else: print(w);
        wf.writelines('\n')

def caltfidf(file,nundic,tag):
    doc = []
    for l in open(file, encoding="UTF-8"):
        ws = l.strip().split()
        for w in ws:
            if w not in w2id:
                w2id[w] = len(w2id)
        doc.append(ws)

    #print(doc)
    tfidf = wd.getTFIDF(doc,nundic,tag,w2id)
    tfidf = sparse.csr_matrix(tfidf)
    # print(tfidf)
    np.save(os.path.join(DATA_DIR, 'CoEmbedding/tfidf-data.npy'), tfidf.data)
    np.save(os.path.join(DATA_DIR, 'CoEmbedding/tfidf-indices.npy'), tfidf.indices)
    np.save(os.path.join(DATA_DIR, 'CoEmbedding/tfidf-indptr.npy'), tfidf.indptr)


def getData():
    #主题模型文档预处理
    #name_str=[]
    line_str=[];
    api_str=[];
    tag_str=[];
    # #读取文件
    df = pd.read_csv("dataset/C12.csv");
    #获得mashup描述和api组成
    for index,row in df.iterrows():
        #name_str.append(row['name'])
        #if(index < 365):
            line_str.append(row['desc'])
            api_str.append(row['APIs'])
            tag_str.append(row['tags'])

    return line_str,api_str,tag_str;
                
if __name__ == '__main__':
    #doc_pt = DATA_DIR+'data.csv'
    #采用NTLK库对CSV预处理
    desc,api,tag = getData()

    # print(desc[0])

    # 接下来将这些英文单词词干化，词干化可以提取不同语态及各种后缀的词干
    dminc_pt = DATA_DIR+'CoEmbedding/desc.txt'
    # C4-3，C8-4，C10，12-5
    min_count = 5

    doc,nundic = wd.get_doc_after_divide(desc)
    #print(doc)
    #计算词频
    wordCount(doc)
    indexFile(doc, dminc_pt, min_count)

    caltfidf(dminc_pt,nundic,tag)
    #print(doc)
    # cutDoc = [];
    # for l in open(dminc_pt,encoding="UTF-8"):
    #     ws = l.strip().split()
    #     cutDoc.append(ws)
    # print(cutDoc)

    #print(len(texts_cleaned[0:5]))
