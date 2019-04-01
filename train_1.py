#coding=utf-8
from __future__ import print_function

import time
import random
import torch.nn as nn
from utility import *
from get_data_1 import *
from model.model import SemanticSimModel
from model.model_1 import MainModel
from model.model_2 import CoAttention
from model.model_3 import ESIMModel

# EMBEDDING_FILEPATH = './word_embedding/sgns.baidubaike.bigram-char'
EMBEDDING_FILEPATH = './word_embedding/sgns.zhihu.bigram-char'
# EMBEDDING_FILEPATH = './word_embedding/sgns.financial.bigram-char'
# EMBEDDING_FILEPATH = './word_embedding/sgns.sogounews.bigram-char'
# EMBEDDING_FILEPATH = './word_embedding/sgns.wiki.bigram-char'
# EMBEDDING_FILEPATH = './word_embedding/sgns.weibo.bigram-char'
# EMBEDDING_FILEPATH = './word_embedding/skipgram.300.txt'

EMBEDDING_DIM = 300     # 词向量维度
HIDDEN_SIZE = 128        # RNN隐藏单元大小
CONTEXT_VEC_SIZE = 512   # 自注意力上下文向量大小
MLP_HID_SIZE = 512       # 多层感知器隐藏层大小
DROPOUT_SIZE = 0.4      # dropout大小
CLASS_SIZE = 1          # 输出类别个数
LEARNING_RATE = 0.002   # learning rate 学习率
WEIGHT_DECAY = 0.0001   # weight decay值大小
KERNEL_NUM = 100        # 每个核的filter个数
KERNEL_SIZES = [2,3,4]  # 核大小
EPOCH = 12               # epoch大小 循环轮数
BATCH_SIZE = 128        # batch_size
PRINT_INTERVAL = 20     # 间隔多少个batch打印一次loss


def preTrainWordEmbedding():
    # 读取预训练的词向量，返回Embedding Dict
    embedding_dict = {}
    f = codecs.open(EMBEDDING_FILEPATH, 'r', encoding='utf-8')
    for line in f:
        line = line.split()
        if len(line) > 0:
            word = line[0]
            if len(line) == EMBEDDING_DIM + 1:
                embed_vec = np.array(line[1:], dtype="float32")
                embedding_dict[word] = embed_vec
    f.close()

    return embedding_dict


def data_package(data, requires_grad=False):
    max_length = 0
    batch = len(data)
    for item in data:
        max_length = max(max_length, len(item))
    return_data = np.zeros((batch, max_length), dtype=int)
    for i in range(batch):
        for word in data[i]:
            index = data[i].index(word)
            if word not in word_dict:
                return_data[i, index] = word_dict['<unk>']
            else:
                return_data[i, index] = word_dict[word]
    return_data = torch.from_numpy(return_data)
    return_data.requires_grad_(requires_grad)

    return return_data

def data_shuffle(s1, s2, labels, seed):
    random.seed(seed)
    random.shuffle(s1)
    random.seed(seed)
    random.shuffle(s2)
    random.seed(seed)
    random.shuffle(labels)

def evaluate(s1, s2, labels):
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
    # s1_hidden = model.init_hidden(BATCH_SIZE)
    # s2_hidden = model.init_hidden(BATCH_SIZE)

    with torch.no_grad():
        for batch_index, start_index in enumerate(range(0, len(s1), BATCH_SIZE)):
            s1_data = data_package(s1[start_index: start_index + BATCH_SIZE], requires_grad=False)
            s2_data = data_package(s2[start_index: start_index + BATCH_SIZE], requires_grad=False)
            label = torch.FloatTensor(labels[start_index: start_index + BATCH_SIZE])
            if use_cuda:
                s1_data = s1_data.cuda()
                s2_data = s2_data.cuda()
                label = label.cuda()

            # s1_hidden = repackage_hidden(s1_hidden)
            # s2_hidden = repackage_hidden(s2_hidden)
            s1_hidden = model.init_hidden(BATCH_SIZE)
            s2_hidden = model.init_hidden(BATCH_SIZE)

            predict = model(s1_data, s2_data, s1_hidden, s2_hidden)
            loss = loss_function(predict, label)
            total_loss += len(s1_data) * loss.data

            p, r, f, a = computeMeasure(predict.view(-1), label.view(-1))
            acc.append(a)
            pre.append(p)
            rec.append(r)
            f1.append(f)

    print('-' * 80)
    print('Precision:', np.mean(pre))
    print('Recall:', np.mean(rec))
    print('F1:', np.mean(f1))
    print('Acc:', np.mean(acc))

    return total_loss / len(s1)

def train(epoch_num, train_s1, train_s2, train_labels):

    model.train(mode=True)

    epoch_start_time = time.time()
    start_time = time.time()
    total_loss = 0
    batch_loss = []

    # s1_hidden = model.init_hidden(BATCH_SIZE)
    # s2_hidden = model.init_hidden(BATCH_SIZE)

    param = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(param, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # optimizer = torch.optim.Adadelta(param, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for batch_index, start_index in enumerate(range(0, len(train_s1), BATCH_SIZE)):
        s1_data = data_package(train_s1[start_index: start_index + BATCH_SIZE], requires_grad=True)
        s2_data = data_package(train_s2[start_index: start_index + BATCH_SIZE], requires_grad=True)
        label = torch.FloatTensor(train_labels[start_index: start_index + BATCH_SIZE])
        if use_cuda:
            s1_data = s1_data.cuda()
            s2_data = s2_data.cuda()
            label = label.cuda()

        model.zero_grad()
        optimizer.zero_grad()

        # s1_hidden = repackage_hidden(s1_hidden)
        # s2_hidden = repackage_hidden(s2_hidden)
        s1_hidden = model.init_hidden(BATCH_SIZE)
        s2_hidden = model.init_hidden(BATCH_SIZE)

        predict = model(s1_data, s2_data, s1_hidden, s2_hidden)

        # print('predict: ', predict[:5])
        # print('label: ', label_batch[:5])

        loss = loss_function(predict, label)

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
                epoch_num, batch_index, len(train_s1) // BATCH_SIZE,
                                    elapsed_time / PRINT_INTERVAL, cur_loss))

            total_loss = 0
            start_time = time.time()

    all_loss.append(np.mean(batch_loss))
    validate_loss = evaluate(val_s1, val_s2, val_labels)

    print('| end of epoch {:2d} | time: {:5.2f}s | valid loss {:5.8f} | '
          .format(epoch, (time.time() - epoch_start_time), validate_loss))
    print('-' * 80)



if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()

    print('>' * 5 + " 正在进行数据生成和预处理 " + '<' * 5)
    preprocessing_start_time = time.time()

    train_s1, train_s2, train_labels, val_s1, val_s2, val_labels, vocab_size, word_dict = dataGeneration()

    # 去除最后一个batch(不足batch_size)
    train_num_batch = len(train_s1) // BATCH_SIZE
    train_s1 = train_s1[: (train_num_batch * BATCH_SIZE)]
    train_s2 = train_s2[: (train_num_batch * BATCH_SIZE)]
    train_labels = train_labels[: (train_num_batch * BATCH_SIZE)]

    val_num_batch = len(val_s1) // BATCH_SIZE
    val_s1 = val_s1[: (val_num_batch * BATCH_SIZE)]
    val_s2 = val_s2[: (val_num_batch * BATCH_SIZE)]
    val_labels = val_labels[: (val_num_batch * BATCH_SIZE)]

    print(' 训练集样例: ', len(train_s1))
    print(' 验证集样例: ', len(val_s1))
    print(' 词表大小: ', vocab_size)
    print('>' * 5 + " 正在读取预训练的词向量 " + '<' * 5)

    embedding_dict = preTrainWordEmbedding()

    print('>' * 5 + " 读取完毕 " + '<' * 5)
    print('>' * 5 + " 预处理所用时间:  " + str(time.time() - preprocessing_start_time) + 's ' + '<' * 5)
    print('>' * 5 + " 模型开始训练： " + '<' * 5)

    #model = SemanticSimModel(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, CONTEXT_VEC_SIZE, MLP_HID_SIZE,
    #                         embedding_dict, word_dict, DROPOUT_SIZE, CLASS_SIZE, use_cuda)
    #model = DRCNModel(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, embedding_dict, word_dict, DROPOUT_SIZE, CLASS_SIZE, use_cuda)
    model = ESIMModel(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, CONTEXT_VEC_SIZE, MLP_HID_SIZE, embedding_dict,
                      word_dict, DROPOUT_SIZE, CLASS_SIZE, use_cuda)
    #model = MainModel(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, CONTEXT_VEC_SIZE, MLP_HID_SIZE, KERNEL_NUM, KERNEL_SIZES,
    #                  embedding_dict, word_dict, DROPOUT_SIZE, CLASS_SIZE, use_cuda)

    if use_cuda:
        model = model.cuda()

    loss_function = nn.BCELoss()

    all_loss = []

    seed = 1000
    for epoch in range(1, EPOCH + 1):
        seed += 10
        data_shuffle(train_s1, train_s2, train_labels, seed)
        train(epoch, train_s1, train_s2, train_labels)


































