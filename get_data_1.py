# coding:utf-8

import torch
import pandas as pd
import numpy as np
import jieba
import codecs
from sklearn.model_selection import train_test_split

# jieba.load_userdict('./data/jieba_dict')

def get_stop_words():
    stop_words_list = []
    with codecs.open('./dict/stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            stop_words_list.append(line)
    return stop_words_list

def remove_stop_words(sentence, stop_words_list):
    new_sentence = []
    for word in sentence:
        if word not in stop_words_list:
            new_sentence.append(word)
    return new_sentence

def get_replace_word_dict():
    replace_word_dict = {}
    with codecs.open('./dict/replace_words.csv', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            if line[0] not in replace_word_dict:
                replace_word_dict[line[0]] = line[1]
    return replace_word_dict

def replace_word(sentence, replace_word_dict):
    '''替换语料中的部分错别字'''
    sentence = sentence.replace(u"花被", u"花呗").replace(u"ma", u"吗").replace(u"花唄", u"花呗") \
        .replace(u"花坝", u"花呗").replace(u"花多上", u"花多少").replace(u"花百", u"花呗").replace(u"花背", u"花呗") \
        .replace(u"节清", u"结清").replace(u"花花", u"花").replace(u"花能", u"花呗能").replace(u"花裁", u"花")\
        .replace(u"花贝", u"花呗").replace(u"花能", u"花呗能").replace(u"蚂蚊", u"蚂蚁").replace(u"蚂蚱", u"蚂蚁")\
        .replace(u"蚂议", u"蚂蚁").replace(u"螞蟻", u"蚂蚁").replace(u"借唄", u"借呗").replace(u"发呗", u"花呗") \
        .replace(u"结呗", u"借呗").replace(u'戒备', u'借呗').replace(u'芝麻', u'').replace(u'压金', u'押金')

    for key, value in replace_word_dict.items():
        sentence = sentence.replace(key, value)
    return sentence

def jieba_cut(sentence):
    '''切成字，并且保留切的词'''
    seg_sent = jieba.cut(sentence, cut_all=False)
    sentence_list = []
    for word in seg_sent:
        if len(word) == 1:
            sentence_list.append(word)
        else:
            for character in word:
                sentence_list.append(character)
            sentence_list.append(word)
    return sentence_list

def word_cut(sentence):
    # 将句子分成单个字
    return [word for word in sentence]

def pre_processing(sentence, replace_word_dict, stop_words_list):
    sentence = replace_word(sentence, replace_word_dict)
    sentence = word_cut(sentence)
    sentence = remove_stop_words(sentence, stop_words_list)
    return  sentence

def dataGeneration():
    train1 = pd.read_csv('./data/atec_nlp_sim_train_add.csv', sep='\t', header=None, encoding='utf-8')
    train1.columns = ['id', 'question1', 'question2', 'is_duplicate']

    train2 = pd.read_csv('./data/atec_nlp_sim_train.csv', sep='\t', header=None, encoding='utf-8')
    train2.columns = ['id', 'question1', 'question2', 'is_duplicate']

    train = train1.append(train2)

    # test = pd.read_csv(INPUT_PATH, sep='\t', header=None, encoding='utf-8')
    # test.columns = ['id', 'question1', 'question2']
    # test = test.fillna(' ')

    replace_word_dict = get_replace_word_dict()
    stop_words_list = get_stop_words()

    # 预处理过程
    train.iloc[:, 1] = train.iloc[:, 1].apply(lambda x: pre_processing(x, replace_word_dict, stop_words_list))
    train.iloc[:, 2] = train.iloc[:, 2].apply(lambda x: pre_processing(x, replace_word_dict, stop_words_list))

    word_dict = {'<pad>':0, '<unk>':1}

    for instance in train.values:
        sentences = instance[1] + instance[2]   # sentences是两个句子中所有字的列表
        for word in sentences:
            if word not in word_dict:
                word_dict[word] = len(word_dict)    # 生成word2index字典

    vocab_size = len(word_dict)

    sentence_pairs = []
    for instance in train.values:
        sentence_pairs.append([instance[1], instance[2]])
    labels = train['is_duplicate'].tolist()
    labels = np.asarray(labels)
    labels = np.reshape(labels, (-1, 1)).tolist()

    '''
    x_dev, y_dev: 开发集(训练集+验证集)
    x_train, y_train: 训练集
    x_val, y_val: 验证集
    x_test, y_test: 测试集
    '''
    x_train, x_val, y_train, y_val = train_test_split(sentence_pairs, labels, test_size=0.1, train_size=0.9,
                                                      random_state=1234, shuffle=True)
    # x_train, x_val, y_train, y_val = train_test_split(x_dev, y_dev, test_size=0.15, train_size=0.85,
    #                                                   shuffle=True)


    s1_train = []
    s2_train = []
    for instance in x_train:
        s1_train.append(instance[0])
        s2_train.append(instance[1])

    s1_val = []
    s2_val = []
    for instance in x_val:
        s1_val.append(instance[0])
        s2_val.append(instance[1])

    # s1_test = []
    # s2_test = []
    # for instance in x_test:
    #     s1_test.append(instance[0])
    #     s2_test.append(instance[1])


    return s1_train, s2_train, y_train, s1_val, s2_val, y_val, vocab_size, word_dict

































