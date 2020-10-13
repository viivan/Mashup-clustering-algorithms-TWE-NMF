# Mashup-clustering-algorithms-TWE-NMF
分为两个文件夹，包含以下算法实现代码：<br>
T+Q：通过TF-IDF将每个Mashup服务描述文档表示成向量形式，进行QT聚类。<br>
LDA+K：通过LDA主题模型对Mashup服务文档进行主题建模得到主题特征。在此基础上，利用k-means算法对主题向量进行聚类。<br>
LDA+API+K：通过Word2Vec 对API描述进行预训练，得到词向量，对于Mashup服务描述中每个单词，在训练好的词向量模型中找出其前3个相似词。最后，将这些相似词合并到Mashup 服务描述文本中，通过LDA主题模型建模后，采用k-means进行聚类。<br>
LDA+wiki+K：通过Word2Vec 对wiki语料库进行预训练，得到词向量，对于Mashup服务描述中每个单词，在训练好的词向量模型中找出其前3个相似词。最后，将这些相似词合并到Mashup 服务描述文本中，通过LDA主题模型建模后，采用k-means进行聚类。<br>
BTM+K：采用针对短文本改进的BTM主题模型对Mashup服务主题建模，随后使用K-means聚类。<br>
GPU-DMM+K：采用结合词嵌入信息的GPU-DMM主题模型对Mashup进行K-means聚类。<br>
CLM+SC: 通过CLM主题模型对Mashup服务主题建模，采用谱聚类的方法对结果进行聚类。<br>
TWE-NMF+SC：我们提出的方法，综合结合词嵌入和服务标签信息，对Mashup服务进行主题建模，采用谱聚类的方法对进行聚类。<br>


其中，k-means和谱聚类都是调用sklearn中的库实现<br>
LDA采用 https://github.com/lda-project/lda  基于Gibbs采样实现<br>
GUPDMM采用https://github.com/NobodyWHU/GPUDMM 的代码<br>
