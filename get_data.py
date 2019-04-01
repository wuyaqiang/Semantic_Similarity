# coding:utf-8

import torch
import pandas as pd
import jieba
import codecs

BATCH_SIZE = 100
SENTENCE_SIZE = 20
# jieba.load_userdict('./dict/important_word')

# def stopwordslist(filepath):
#     stopwords = [line.strip() for line in codecs.open(filepath, 'r', encoding='utf8').readlines()]
#     return stopwords
# stopwords = stopwordslist('dict/stop_word')

# def fenci(sentence):
#     seg_list = jieba.cut(sentence, cut_all=False)
#     outseg = ''
#     for word in seg_list:
#         if word not in stopwords:
#             outseg += word
#             outseg += ' '
#     return outseg

def replaceWord(sentence):
    # 替换语料中的部分错别字
    sentence = sentence.replace(u"花被", u"花呗").replace(u"ma", u"吗").replace(u"花唄", u"花呗")\
        .replace(u"花坝", u"花呗").replace(u"花多上", u"花多少").replace(u"花百", u"花呗").replace(u"花背", u"花呗")\
        .replace(u"节清", u"结清").replace(u"花花", u"花").replace(u"花能", u"花呗能")\
        .replace(u"花裁", u"花").replace(u"花贝", u"花呗").replace(u"花能", u"花呗能")\
        .replace(u"蚂蚊", u"蚂蚁").replace(u"蚂蚱", u"蚂蚁").replace(u"蚂议", u"蚂蚁")\
        .replace(u"螞蟻", u"蚂蚁").replace(u"借唄", u"借呗").replace(u"发呗", u"花呗")\
        .replace(u"结呗", u"借呗")
    return sentence

def wordCut(sentence):
    # 将句子分成单个字
    return [word for word in sentence]

def dataGeneration():
    train1 = pd.read_csv('./data/atec_nlp_sim_train_add.csv', sep='\t', header=None, encoding='utf-8')
    train1.columns = ['id', 'question1', 'question2', 'is_duplicate']

    train2 = pd.read_csv('./data/atec_nlp_sim_train.csv', sep='\t', header=None, encoding='utf-8')
    train2.columns = ['id', 'question1', 'question2', 'is_duplicate']

    train = train1.append(train2)

    # test = pd.read_csv(INPUT_PATH, sep='\t', header=None, encoding='utf-8')
    # test.columns = ['id', 'question1', 'question2']
    # test = test.fillna(' ')

    train.iloc[:, 1] = train.iloc[:, 1].apply(lambda x: replaceWord(x))
    train.iloc[:, 2] = train.iloc[:, 2].apply(lambda x: replaceWord(x))

    train.iloc[:, 1] = train.iloc[:, 1].apply(lambda x: wordCut(x))
    train.iloc[:, 2] = train.iloc[:, 2].apply(lambda x: wordCut(x))

    word_dict = {}
    max_length = 0
    num_instance = 0

    for instance in train.values:
        num_instance += 1
        max_length = max(max_length, len(instance[1]), len(instance[2]))
        sentences = instance[1] + instance[2]   # sentences是两个句子中所有字的列表
        for word in sentences:
            if word not in word_dict:
                word_dict[word] = len(word_dict) + 1

    vocab_size = len(word_dict)
    sentence_1 = torch.LongTensor(num_instance, max_length).fill_( 0 )
    sentence_2 = torch.LongTensor(num_instance, max_length).fill_( 0 )
    s1_lengths = []
    s2_lengths = []
    labels = torch.FloatTensor(num_instance, 1).fill_( 0 )

    for index, instance in enumerate(train.values):
        labels[index] = instance[3]
        s1_lengths.append( len(instance[1]) )
        s2_lengths.append( len(instance[2]) )
        for i, word in enumerate(instance[1]):
            sentence_1[index][i] = word_dict[word]
        for i, word in enumerate(instance[2]):
            sentence_2[index][i] = word_dict[word]

    divide_point = int(num_instance * 0.9)

    train_s1 = sentence_1[ : divide_point]
    train_s2 = sentence_2[ : divide_point]
    train_s1_lengths = s1_lengths[ : divide_point]
    train_s2_lengths = s2_lengths[ : divide_point]
    train_labels = labels[ : divide_point]

    val_s1 = sentence_1[divide_point : ]
    val_s2 = sentence_2[divide_point : ]
    val_s1_lengths = s1_lengths[divide_point : ]
    val_s2_lengths = s2_lengths[divide_point : ]
    val_labels = labels[divide_point : ]

    return train_s1, train_s2, train_s1_lengths, train_s2_lengths, train_labels, \
           val_s1, val_s2, val_s1_lengths, val_s2_lengths, val_labels, \
           vocab_size, word_dict


































