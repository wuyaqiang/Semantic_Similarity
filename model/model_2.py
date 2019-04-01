#coding=utf-8
from __future__ import print_function

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

'''Densely-connected Recurrent and Co-attentive neural Network'''
class CoAttention(nn.Module):
    def __init__(self, global_state_dim, hidden_size, use_cuda):
        super(CoAttention, self).__init__()

        self.lstm = nn.LSTM(global_state_dim, hidden_size, bidirectional=True, batch_first=True)
        self.CosineSim = nn.CosineSimilarity(dim=3)
        self.softmax = nn.Softmax(dim=2)
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

    def forward(self, s1_global_state, s2_global_state, s1_hidden, s2_hidden):

        s1_lstm_output, s1_hidden = self.lstm(s1_global_state, s1_hidden)
        s2_lstm_output, s2_hidden = self.lstm(s2_global_state, s2_hidden)

        s1_global_state = torch.cat((s1_global_state, s1_lstm_output), 2)
        s2_global_state = torch.cat((s2_global_state, s2_lstm_output), 2)

        s1_expanded = s1_lstm_output.unsqueeze(2).expand(s1_lstm_output.size()[0], s1_lstm_output.size()[1],
                                                         s2_lstm_output.size()[1], s1_lstm_output.size()[2])
        s2_expanded = s2_lstm_output.unsqueeze(1).expand(s2_lstm_output.size()[0], s1_lstm_output.size()[1],
                                                         s2_lstm_output.size()[1], s2_lstm_output.size()[2])

        cosine_sim = self.CosineSim(s1_expanded, s2_expanded)
        s1_attentive = torch.matmul(self.softmax(cosine_sim), s2_lstm_output)
        s2_attentive = torch.matmul(self.softmax(cosine_sim.transpose(1,2)), s1_lstm_output)

        s1_global_state = torch.cat((s1_global_state, s1_attentive), 2)
        s2_global_state = torch.cat((s2_global_state, s2_attentive), 2)

        return s1_global_state, s2_global_state, s1_hidden, s2_hidden


class DRCNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, embedding_dict, word_dict, dropout_size, class_size, use_cuda):
        super(DRCNModel, self).__init__()

        self.use_cuda = use_cuda
        self.word_dict = word_dict
        self.embedding_dict = embedding_dict
        self.hidden_size = hidden_size

        # Embedding layer
        self.embedding2 = nn.Embedding(vocab_size, embedding_dim)   # Trainable
        self.init_pretrained_embedding()
        self.embedding1 = nn.Embedding.from_pretrained(self.embedding2.weight.data, freeze=True)   # Fixed
        # non linearity function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dense1 = CoAttention(embedding_dim * 2, hidden_size, use_cuda)
        self.linear1 = nn.Linear(embedding_dim * 2 + hidden_size * 4, hidden_size * 4)

        self.dense2 = CoAttention(hidden_size * 4, hidden_size, use_cuda)
        self.linear2 = nn.Linear(hidden_size * 8, hidden_size * 4)

        self.dense3 = CoAttention(hidden_size * 4, hidden_size, use_cuda)
        self.linear3 = nn.Linear(hidden_size * 8, hidden_size * 4)
        # mlp layer
        self.mlp = nn.Linear(hidden_size * 20, hidden_size * 10)
        # dropout layer
        self.dropout = nn.Dropout(dropout_size)
        # output layer
        self.output = nn.Linear(hidden_size * 10, class_size)
        # initialize the weight
        self.init_weight()


    def forward(self, s1, s2, s1_hidden, s2_hidden):
        s1_pre_trained = self.embedding1(s1)
        s2_pre_trained = self.embedding1(s2)

        s1_embeded = self.embedding2(s1)
        s2_embeded = self.embedding2(s2)

        s1_embeded = torch.cat((s1_embeded, s1_pre_trained), 2)
        s2_embeded = torch.cat((s2_embeded, s2_pre_trained), 2)

        s1_global_state, s2_global_state, s1_hidden, s2_hidden = self.dense1(s1_embeded, s2_embeded,
                                                                             s1_hidden, s2_hidden)
        s1_global_state = self.relu(self.linear1(s1_global_state))
        s2_global_state = self.relu(self.linear1(s2_global_state))

        s1_global_state, s2_global_state, s1_hidden, s2_hidden = self.dense2(s1_global_state, s2_global_state,
                                                                             s1_hidden, s2_hidden)
        s1_global_state = self.relu(self.linear2(s1_global_state))
        s2_global_state = self.relu(self.linear2(s2_global_state))

        s1_global_state, s2_global_state, s1_hidden, s2_hidden = self.dense3(s1_global_state, s2_global_state,
                                                                             s1_hidden, s2_hidden)
        s1_global_state = self.relu(self.linear3(s1_global_state))
        s2_global_state = self.relu(self.linear3(s2_global_state))

        s1_global_state, _ = torch.max(s1_global_state, 1)
        s2_global_state, _ = torch.max(s2_global_state, 1)

        '''merged = [ s1 ; s2 ; s1 + s2 ; s1 - s2 ; |s1 - s2| ] '''
        merge_add = torch.add(s1_global_state, s2_global_state)
        neg_s2 = torch.neg(s2_global_state)
        merge_minus = torch.add(s1_global_state, neg_s2)
        merge_abs = torch.abs(merge_minus)
        merged = torch.cat((s1_global_state, s2_global_state, merge_add, merge_minus, merge_abs), 1)

        if self.use_cuda:
            merged = merged.cuda()

        merged = self.dropout(merged)
        output_mlp = self.mlp(merged)
        output_mlp = self.relu(output_mlp)
        # output_mlp = self.dropout(output_mlp)

        output = self.output(output_mlp)
        output = self.sigmoid(output)
        return output


    def init_weight(self):
        nn.init.xavier_normal_(self.linear1.weight.data)
        nn.init.xavier_normal_(self.linear2.weight.data)
        nn.init.xavier_normal_(self.linear3.weight.data)

        nn.init.xavier_normal_(self.mlp.weight.data)
        # self.mlp.weight.data.uniform_(-init_range, init_range)
        self.mlp.bias.data.fill_(0)

        nn.init.xavier_normal_(self.output.weight.data)
        # self.output.weight.data.uniform_(-init_range, init_range)
        self.output.bias.data.fill_(0)

    def init_pretrained_embedding(self):
        # 初始化预训练的词向量矩阵
        # nn.init.xavier_normal_(self.embedding1.weight.data)
        nn.init.xavier_normal_(self.embedding2.weight.data)

        # self.embedding1.weight.data[self.word_dict['<pad>']].fill_(0)
        self.embedding2.weight.data[self.word_dict['<pad>']].fill_(0)

        # self.embedding1.weight.requires_grad = False

        loaded_count = 0
        for word in self.word_dict:
            if word not in self.embedding_dict:
                continue
            real_id = self.word_dict[word]
            # self.embedding1.weight.data[real_id] = torch.from_numpy(self.embedding_dict[word]).view(-1)
            self.embedding2.weight.data[real_id] = torch.from_numpy(self.embedding_dict[word]).view(-1)
            loaded_count += 1
        print(' %d words from pre-trained word vectors loaded.' % loaded_count)

    def init_hidden(self, batch_size):
        '''
        初始化 LSTM 隐藏单元的权值
        '''
        hidden = torch.zeros((2, batch_size, self.hidden_size), requires_grad=True)
        cell = torch.zeros((2, batch_size, self.hidden_size), requires_grad=True)
        if self.use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return (hidden, cell)
        # weight = next(self.parameters()).data
        # return (torch.FloatTensor(weight.new(2, batch_size, self.hidden_size).fill_(0)),
        #         torch.FloatTensor(weight.new(2, batch_size, self.hidden_size).fill_(0)))

















