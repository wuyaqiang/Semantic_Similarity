#coding=utf-8
from __future__ import print_function

import time
import torch.nn as nn
import numpy as np
import codecs

from utility import *
from get_data import dataGeneration
from model.model import SemanticSimModel

EMBEDDING_FILEPATH = './data/sgns.baidubaike.bigram-char'

EMBEDDING_DIM = 300     # 词向量维度
HIDDEN_SIZE = 128        # RNN隐藏单元大小
CONTEXT_VEC_SIZE = 128   # 自注意力上下文向量大小
MLP_HID_SIZE = 128       # 多层感知器隐藏层大小
DROPOUT_SIZE = 0.2      # dropout大小
CLASS_SIZE = 1          # 输出类别个数
LEARNING_RATE = 0.002   # learning rate 学习率
EPOCH = 5               # epoch大小 循环轮数
BATCH_SIZE = 200        # batch_size
PRINT_INTERVAL = 10     # 间隔多少个batch打印一次loss

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

def preTrainWordEmbedding():
    # 读取预训练的词向量，返回Embedding Dict
    embedding_dict = {}
    f = codecs.open(EMBEDDING_FILEPATH, 'r', encoding='utf-8')
    for line in f:
        line = line.split()
        word = line[0]
        if len(line) == EMBEDDING_DIM + 1:
            embed_vec = np.array(line[1:], dtype="float32")
            embedding_dict[word] = embed_vec
    f.close()

    return embedding_dict

def evaluate(s1, s2, s1_lengths, s2_lengths, labels):
    '''
    计算precision,recall,以及F1值
    '''
    model.eval()    # switch to evaluate mode

    acc = []
    pre = []
    rec = []
    f1 = []

    total_loss = 0
    loss_function = nn.BCELoss()
    s1_hidden = model.init_hidden(BATCH_SIZE)
    s2_hidden = model.init_hidden(BATCH_SIZE)

    with torch.no_grad():
        for batch_index, start_index in enumerate(range(0, s1.size(0) - 1, BATCH_SIZE)):
            s1_data = s1[start_index: start_index + BATCH_SIZE, :]
            s2_data = s2[start_index: start_index + BATCH_SIZE, :]
            s1_length = s1_lengths[start_index: start_index + BATCH_SIZE]
            s2_length = s2_lengths[start_index: start_index + BATCH_SIZE]
            label_batch = labels[start_index: start_index + BATCH_SIZE]

            predict = model(s1_data, s2_data, s1_length, s2_length, s1_hidden, s2_hidden)
            loss = loss_function(predict, label_batch)
            total_loss += loss.item()

            s1_hidden = repackage_hidden(s1_hidden)
            s2_hidden = repackage_hidden(s2_hidden)

            p, r, f, a = computeMeasure(predict.view(-1), label_batch.view(-1))
            acc.append(a)
            pre.append(p)
            rec.append(r)
            f1.append(f)

    print('Precision:', np.mean(pre))
    print('Recall:', np.mean(rec))
    print('F1:', np.mean(f1))
    print('Acc:', np.mean(acc))

    return total_loss / len(s1)


if __name__ == '__main__':

    print('>' * 5 + "预处理阶段: ")

    train_s1, train_s2, train_s1_lengths, train_s2_lengths, train_labels, \
    val_s1, val_s2, val_s1_lengths, val_s2_lengths, val_labels, \
    vocab_size, word_dict = dataGeneration()

    # 临时处理,去除最后一个batch(不足batch_size)
    train_s1 = train_s1[: 92000]
    train_s2 = train_s2[: 92000]
    train_s1_lengths = train_s1_lengths[: 92000]
    train_s2_lengths = train_s2_lengths[: 92000]
    train_labels = train_labels[: 92000]
    val_s1 = val_s1[: 10000]
    val_s2 = val_s2[: 10000]
    val_s1_lengths = val_s1_lengths[: 10000]
    val_s2_lengths = val_s2_lengths[: 10000]
    val_labels = val_labels[: 10000]

    print('训练样例总数: ', len(train_s1))
    print('验证样例总数: ', len(val_s1))
    print('词表索引词典样例: ', word_dict.items()[:5])
    print('词表大小: ', vocab_size)
    print('类标样例: ', train_labels[:5])
    print('向量化的句子样例: ', train_s1[:5])

    print('>' * 5 + "预处理阶段完成")
    print('\n')

    embedding_dict = preTrainWordEmbedding()

    print("模型开始训练" + '.' * 5)

    model = SemanticSimModel(vocab_size + 1, EMBEDDING_DIM, HIDDEN_SIZE, CONTEXT_VEC_SIZE, MLP_HID_SIZE,
                             embedding_dict, DROPOUT_SIZE, CLASS_SIZE)

    loss_function = nn.BCELoss()

    all_loss = []

    for epoch in range(1, EPOCH + 1):

        model.train()

        epoch_start_time = time.time()
        start_time = time.time()
        total_loss = 0
        batch_loss = []

        s1_hidden = model.init_hidden(BATCH_SIZE)
        s2_hidden = model.init_hidden(BATCH_SIZE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for batch_index, start_index in enumerate( range(0, train_s1.size(0) - 1, BATCH_SIZE) ):
            s1_data = train_s1[start_index : start_index + BATCH_SIZE, :]
            s2_data = train_s2[start_index : start_index + BATCH_SIZE, :]
            s1_length = train_s1_lengths[start_index : start_index + BATCH_SIZE]
            s2_length = train_s2_lengths[start_index : start_index + BATCH_SIZE]
            label_batch = train_labels[start_index : start_index + BATCH_SIZE]

            model.zero_grad()
            optimizer.zero_grad()

            s1_hidden = repackage_hidden(s1_hidden)
            s2_hidden = repackage_hidden(s2_hidden)

            predict = model(s1_data, s2_data, s1_length, s2_length, s1_hidden, s2_hidden)

            # print('predict: ', predict[:5])
            # print('label: ', label_batch[:5])

            loss = loss_function(predict, label_batch)

            loss.backward()
            # Gradient clipping in case of gradient explosion
            # torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            batch_loss.append(loss.item())

            if batch_index % PRINT_INTERVAL == 0 and batch_index > 0:
                cur_loss = total_loss / PRINT_INTERVAL
                elapsed_time = time.time() - start_time
                print('| epoch {:2d} | {:4d}/{:3d} batches | {:5.2f} s/batch | '
                      'loss {:5.4f} |'.format(
                    epoch, batch_index, len(train_s1) // BATCH_SIZE,
                    elapsed_time / PRINT_INTERVAL, cur_loss))

                total_loss = 0
                start_time = time.time()

        all_loss.append(np.mean(batch_loss))
        validate_loss = evaluate(val_s1, val_s2, val_s1_lengths, val_s2_lengths, val_labels)

        print('-' * 80)
        print('| end of epoch {:2d} | time: {:5.2f}s | valid loss {:5.4f} | '
              .format(epoch, (time.time() - epoch_start_time), validate_loss))
        print('-' * 80)

































