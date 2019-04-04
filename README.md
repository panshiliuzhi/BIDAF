# BIDAF
implement Machine Reading Comprehension Model introduced by [Bi-directional Attention Flow for Machine Comprehension][1].

# [QANET][2]
### QANet.py文件说明
* __init__:初始化一些参数和配置文件

* build_graph:建立计算图， 主要调用下面的搭模型的方法
* placeholders :设置tensorflow 模型占位符，也就是传入的参数，不是模型参数，主要是原文序列（batch_size, passage_length, embedding_size），问题序列（batch_size, question_length, embedding_size），原文长度序列（shape = [batch_size],每一个值为每一个样本的长度），问题长度序列（batch_size）， start(shape = [batch_size], 一批样本对应的开始位置)， end同样，dropout_keep_prob：是dropout参数
* word_embedding：之前是加载自己预训练的词向量，现在是用得elmo,参考：[如何将ELMo词向量用于中文][3]
* encoder :对应原论文中：Embedding Encoder Layer，实现了其中的position Encoding, Layernorm, depthwise separable conv, Self-attention,主要方法在blocks.py文件中，分别对问题和原文进行encode,参数进行共享
* attention_flow：使用的BIDAF的context-query attention和query-context attention部门，对应于attention.py中的content_attention
* model_encoder:方法和encoder中的一模一样，调用的也是一样的方法
* output：使用了一个全连接层就是论文中的W_1[M0;M1], 和W2[M0; M2]
* compute_loss：求loss
* train_one_epoch:每一轮train的方法
* train：train的主方法
* evaluate：每跑完50轮评价一下模型，如果高于上一轮的效果就保存当前模型的参数
* dev_content_answer：加载验证集

### run.py文件说明

主要是设置超参数以及加载数据集
  [1]: https://arxiv.org/abs/1611.01603
  [2]: https://arxiv.org/pdf/1804.09541.pdf
  [3]: http://www.linzehui.me/2018/08/12/%E7%A2%8E%E7%89%87%E7%9F%A5%E8%AF%86/%E5%A6%82%E4%BD%95%E5%B0%86ELMo%E8%AF%8D%E5%90%91%E9%87%8F%E7%94%A8%E4%BA%8E%E4%B8%AD%E6%96%87/