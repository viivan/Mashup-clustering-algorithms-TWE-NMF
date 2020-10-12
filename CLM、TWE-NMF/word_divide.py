import re
import nltk
import numpy as np
import math
from gensim import corpora
from gensim import models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import functools

"""
文档预处理
分句，分词，提取词语主干
"""

# 获取英语的停止词
stop = stopwords.words("english")
# stop.append("ve")
stop.append("etc")
stop.append("n't")
stop.append("api")
stop.append("apis")
stop.append("apis")
stop.append("mashup")

# 获取wordNet词形还原帮助类
lemma = WordNetLemmatizer()
#名词
noun_List = []
#c
dict  = {};
dictC = {}

def isSymbol(word):
    return bool(re.match(r'[^\w]', word))

# \b\d+\b 纯数字
def hasNumber(word):
    return bool(re.search(r'\d', word))


def check(word):
    if isSymbol(word):
        return False
    if hasNumber(word):
        return False
    return True

#词性标注
def wordMark(doc):

    #分词
    #print(doc)
    words = nltk.word_tokenize(doc)
     #词性标注
    pos_tags = nltk.pos_tag(words)
    for word, pos in pos_tags:
        word = word.lower()
        dict[word] = pos
    #print(dict)
    #return dict

#计算相似度
def calSim(doc, tag_str):

    sim_dict = {}  # 名词平均相似度词典
    sim_list = []  # 存放各描述文本名词平均相似度的数组
    #print(noun_list1)
    for (temp_set, tag) in zip(doc, tag_str):
        for i in temp_set:
            # 初始化初始相似度
            simi = 0
            tsim = 0
            tagi = tag.split(",")

            for t in tagi:
                try:
                    t = lemma.lemmatize(t.strip().lower(), pos=wordnet.NOUN)
                    senst = wordnet.synset(t + '.n.1')
                    sensi = wordnet.synset(i + '.n.1')
                    sens_path = sensi.path_similarity(senst)
                    if (tsim < sens_path):
                        tsim = sens_path
                except:
                    continue
            for j in temp_set:
                if i == j:
                    continue
                try:
                    sensi = wordnet.synset(i + '.n.1')
                    sensj = wordnet.synset(j + '.n.1')
                    simi += sensi.path_similarity(sensj) / (len(temp_set) - 1)
                except:
                    continue
            w = 0.5
            sim_dict[i] = simi * w + tsim * (1 - w)
            if sim_dict[i] > 1 : print(i)
        sim_list.append(sim_dict)
        sim_dict = {}
    # print(sim_list)
    return sim_list

def calWeight(word_list,noun_list1,sim_list,w2id):
    tfidf_vec = []
    # 赋给语料库中每个词(不重复的词)一个整数id
    # print(noun_List)

    dictionary = corpora.Dictionary(word_list)  # 建立词典
    # for i in dictionary:
    #       print(dictionary[i])
    # print('price' in noun_list1)

    new_corpus = [dictionary.doc2bow(text) for text in word_list]
    # 创建权重矩阵
    wtfidf_vec = np.zeros((len(word_list), len(dictionary)))
    #训练tf-idf值
    tfidf = models.TfidfModel(new_corpus)
    #print(tfidf)
    # tfidf.save("my_model.tfidf")
    # 载入tfidf模型
    # tfidf = models.TfidfModel.load("my_model.tfidf")
    for i in range(len(word_list)):
        string_bow = dictionary.doc2bow(word_list[i])
        string_tfidf = tfidf[string_bow]  # 取得每个词的tfidf值(以元组的形式)
        #print(string_tfidf)
        #string_tfidf = sorted(string_tfidf, key=lambda item: item[1], reverse=True)  # 对元组的tfidf权值按降序进行排序

        tfidf_vec.append(string_tfidf)
    # print(tfidf_vec)

    #标记文档ID
    # for i in dictionary:
    #     print(dictionary[i])
    pos = 0
    # 遍历tfidf_vec
    for (tfidf_list,sim_dic) in zip(tfidf_vec,sim_list):
        # print(tfidf_list)
        for tfidf in tfidf_list:
            # print(dictionary[tfidf[0]])
            if dictionary[tfidf[0]] in noun_list1:
                ti = tfidf[1]
                # print((dictionary[tfidf[0]],ti))
                word_sim = sim_dic[dictionary[tfidf[0]]]
                #print(ti)
                wtfidf_vec[pos][w2id[dictionary[tfidf[0]]]] = ti / (1 - word_sim)
            else:
                wtfidf_vec[pos][w2id[dictionary[tfidf[0]]]] = tfidf[1]
        pos += 1
    return wtfidf_vec
#处理特殊格式单词
def divide(doc):
    # 对每个文档进行，分句，分词，过滤
    # doc为单文档字符串
    # 获取词性标注
    #print(dict)
    sentences = nltk.sent_tokenize(doc)
    #print(sentences)
    words = []
    for sentence in sentences:
        word_list = nltk.word_tokenize(sentence)
        #print(word_list)
        for word in word_list:
            #temp = word
            word = word.lower()
            if check(word) and not (word in stop):
                words.append(word)

    return words
def divideC(doc):
    # 对每个文档进行，分句，分词，过滤,提取词干
    # doc为单文档字符串
    # 获取词性标注
    #print(dict)
    sentences = nltk.sent_tokenize(doc)
    #print(sentences)
    words = []
    for sentence in sentences:
        word_list = nltk.word_tokenize(sentence)
        #print(word_list)
        for word in word_list:
            #temp = word
            word = word.lower()
            if check(word) and not (word in stop):
                # 归一形态
                #获取词根
                word_pos = dict[word]
                if word_pos.startswith('J'):
                    lem = lemma.lemmatize(word, pos=wordnet.ADJ)
                elif word_pos.startswith('V'):
                    lem = lemma.lemmatize(word, pos=wordnet.VERB)
                elif word_pos.startswith('N'):
                    lem = lemma.lemmatize(word, pos=wordnet.NOUN)
                elif word_pos.startswith('R'):
                    lem = lemma.lemmatize(word, pos=wordnet.ADV)
                else:
                    lem = lemma.lemmatize(word, pos=wordnet.NOUN)
                #lem = wordnet.morphy(word)

                if lem is None:
                     words.append(word)
                     dictC[word] = word_pos
                     #标记名词
                     if (word_pos.startswith('N')):
                         if  not word  in noun_List:
                            noun_List.append(word)
                else:
                     words.append(lem)
                     dictC[lem] = word_pos

                     if (word_pos.startswith('N')):
                         if  not lem  in noun_List:
                             noun_List.append(lem)

    return words
def getTFIDF(doc,noudic,tag,w2id):
    sim = calSim(doc, tag)
    return calWeight(doc, noudic, sim,w2id)

def get_doc_after_divide(doc):
    # 获取完整处理后的语料信息，返回为dictionary
    # doc为为文档字典，未分词
    # stop.append(name)
    #wordMark(doc)
    dic = {}
    #print(wordnet.synsets("e-Commerce"))
    for k in doc:
        document = k
        words = divide(document)
        doc_word = " ".join(words)
        if(k in dic.keys()):
            print(k)
        dic[k] = doc_word
    #print(len(dic))
    for k in doc:
        document = dic[k]
        document = document.replace("/"," ").replace("-based"," ")
        wordMark(document)
        words = divideC(document)
        doc_word = " ".join(words)
        dic[k] = doc_word
    return dic.values(),noun_List