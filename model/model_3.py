#coding=utf-8
from __future__ import print_function

import numpy as np
import torch
from torch import nn

'''The Enhanced Sequential Inference Model (ESIM)'''
class ESIMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, context_vec_size, mlp_hid_size,
                 embedding_dict, word_dict, dropout_size, class_size, use_cuda):
        super(ESIMModel, self).__init__()

        self.use_cuda = use_cuda
        self.word_dict = word_dict
        self.embedding_dict = embedding_dict
        self.hidden_size = hidden_size
        self.context_vec_size = context_vec_size
        self.mlp_hid_size = mlp_hid_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # bi-LSTM layer
        self.lstm1 = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size * 4, hidden_size, bidirectional=True, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(dropout_size)
        # mlp layer
        self.mlp1 = nn.Linear(hidden_size * 8, mlp_hid_size)
        self.mlp2 = nn.Linear(hidden_size * 2, mlp_hid_size)
        # output layer
        self.output = nn.Linear(mlp_hid_size, class_size)
        # normalization layer
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size * 2, eps=1e-05, elementwise_affine=True)
        # non linearity function
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        # initialize the weight
        self.init_weight()
        self.init_pretrained_embedding()

    def forward(self, s1, s2, s1_hidden, s2_hidden):
        s1_hidden_init = s1_hidden
        s2_hidden_init = s2_hidden

        s1_embeded = self.embedding(s1)
        s2_embeded = self.embedding(s2)

        s1_output1, s1_hidden = self.lstm1(s1_embeded, s1_hidden)
        s2_output1, s2_hidden = self.lstm1(s2_embeded, s2_hidden)
        s1_output1 = self.layer_norm(s1_output1)
        s2_output1 = self.layer_norm(s2_output1)

        weight_matrix = torch.matmul(s1_output1, s2_output1.transpose(1,2).contiguous())    # (hs, len1, len2)
        s1_attentive = torch.matmul(self.softmax(weight_matrix), s2_output1)
        s2_attentive = torch.matmul(self.softmax(weight_matrix.transpose(1,2).contiguous()), s1_output1)

        weight_matrix_1 = torch.matmul(s1_attentive, s2_attentive.transpose(1,2).contiguous())
        s1_attentive_1 = torch.matmul(self.softmax(weight_matrix_1), s2_attentive)
        s2_attentive_1 = torch.matmul(self.softmax(weight_matrix_1.transpose(1,2).contiguous()), s1_attentive)

        # s1_merged = torch.cat((s1_output1, s1_attentive), 2)
        # s2_merged = torch.cat((s2_output1, s2_attentive), 2)
        # s1_output2, s1_hidden_init = self.lstm2(s1_merged, s1_hidden_init)
        # s2_output2, s2_hidden_init = self.lstm2(s2_merged, s2_hidden_init)
        #s1_output2 = self.layer_norm(s1_output2)
        #s2_output2 = self.layer_norm(s2_output2)

        s1_ave = torch.mean(s1_attentive_1, 1, keepdim=False)
        s2_ave = torch.mean(s2_attentive_1, 1, keepdim=False)
        s1_max, _ = torch.max(s1_attentive_1, 1)
        s2_max, _ = torch.max(s2_attentive_1, 1)

        s1_merged = torch.cat((s1_ave, s1_max), 1)
        s2_merged = torch.cat((s2_ave, s2_max), 1)

        # merged = torch.cat((s1_ave, s1_max, s2_ave, s2_max), 1)
        merge_add = torch.add(s1_merged, s2_merged)
        neg_s2 = torch.neg(s2_merged)
        merge_minus = torch.add(s1_merged, neg_s2)
        merge_minus = torch.pow(merge_minus, 2)
        merged = torch.cat((merge_add, merge_minus), 1)

        if self.use_cuda:
            merged = merged.cuda()

        # merged = self.dropout(merged)
        output = self.relu(self.mlp1(merged))
        # output_mlp2 = self.relu(self.mlp2(output_mlp1))
        # output_mlp = self.dropout(output_mlp)

        output = self.output(output)
        output = self.sigmoid(output)
        return output

    def init_weight(self):
        nn.init.xavier_uniform_(self.mlp1.weight.data)
        nn.init.xavier_uniform_(self.mlp2.weight.data)
        # self.mlp.weight.data.uniform_(-init_range, init_range)
        self.mlp1.bias.data.fill_(0.01)
        self.mlp2.bias.data.fill_(0.01)

        nn.init.xavier_uniform_(self.output.weight.data)
        # self.output.weight.data.uniform_(-init_range, init_range)
        self.output.bias.data.fill_(0.01)

    def init_pretrained_embedding(self):
        # 初始化预训练的词向量矩阵
        nn.init.xavier_uniform_(self.embedding.weight.data)
        self.embedding.weight.data[self.word_dict['<pad>']].fill_(0)
        loaded_count = 0
        for word in self.word_dict:
            if word not in self.embedding_dict:
                continue
            real_id = self.word_dict[word]
            self.embedding.weight.data[real_id] = torch.from_numpy(self.embedding_dict[word]).view(-1)
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