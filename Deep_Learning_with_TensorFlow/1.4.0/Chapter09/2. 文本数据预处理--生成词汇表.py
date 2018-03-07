
# coding: utf-8

# In[6]:


import codecs
import collections
from operator import itemgetter


# #### 1. 设置参数。

# In[19]:


MODE = "PTB"    # 将MODE设置为"PTB", "TRANSLATE_EN", "TRANSLATE_ZH"之一。

if MODE == "PTB":             # PTB数据处理
    RAW_DATA = "../../datasets/PTB_data/ptb.train.txt"  # 训练集数据文件
    VOCAB_OUTPUT = "ptb.vocab"                         # 输出的词汇表文件
elif MODE == "TRANSLATE_ZH":  # 翻译语料的中文部分
    RAW_DATA = "../../datasets/TED_data/train.txt.zh"
    VOCAB_OUTPUT = "zh.vocab"
    VOCAB_SIZE = 4000
elif MODE == "TRANSLATE_EN":  # 翻译语料的英文部分
    RAW_DATA = "../../datasets/TED_data/train.txt.en"
    VOCAB_OUTPUT = "en.vocab"
    VOCAB_SIZE = 10000


# #### 2.对单词按词频排序。

# In[20]:


counter = collections.Counter()
with codecs.open(RAW_DATA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

# 按词频顺序对单词进行排序。
sorted_word_to_cnt = sorted(
    counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]


# #### 3.插入特殊符号。

# In[21]:


if MODE == "PTB":
    # 稍后我们需要在文本换行处加入句子结束符"<eos>"，这里预先将其加入词汇表。
    sorted_words = ["<eos>"] + sorted_words
elif MODE in ["TRANSLATE_EN", "TRANSLATE_ZH"]:
    # 在9.3.2小节处理机器翻译数据时，除了"<eos>"以外，还需要将"<unk>"和句子起始符
    # "<sos>"加入词汇表，并从词汇表中删除低频词汇。
    sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
    if len(sorted_words) > VOCAB_SIZE:
        sorted_words = sorted_words[:VOCAB_SIZE]


# #### 4.保存词汇表文件。

# In[22]:


with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + "\n")

