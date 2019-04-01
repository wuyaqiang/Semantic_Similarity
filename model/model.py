#coding=utf-8
from __future__ import print_function

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SemanticSimModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, context_vec_size, mlp_hid_size,
                 embedding_matrix, dropout_size, class_size):
        super(SemanticSimModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # bi-LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(dropout_size)
        # self-attention layer
        self.S1 = nn.Linear(hidden_size * 2, context_vec_size, bias=False)
        self.S2 = nn.Linear(context_vec_size, 1, bias=False)
        # mlp layer
        self.mlp = nn.Linear(hidden_size * 4, mlp_hid_size)
        # output layer
        self.output = nn.Linear(mlp_hid_size, class_size)
        self.sigmoid = nn.Sigmoid()
        # initialize the weight
        self.init_weight()
        self.init_pretrained_embedding(embedding_matrix)

        self.hidden_size = hidden_size
        self.context_vec_size = context_vec_size
        self.mlp_hid_size = mlp_hid_size

    def init_weight(self, init_range=0.1):
        self.S1.weight.data.uniform_(-init_range, init_range)
        self.S2.weight.data.uniform_(-init_range, init_range)

        self.mlp.weight.data.uniform_(-init_range, init_range)
        self.mlp.bias.data.fill_(0)

        self.output.weight.data.uniform_(-init_range, init_range)
        self.output.bias.data.fill_(0)

    def init_pretrained_embedding(self, embedding_matrix):
        # 初始化预训练的词向量矩阵
        self.embedding.weight.data = torch.from_numpy(embedding_matrix).float()

    def init_hidden(self, batch_size):
        '''
        初始化 LSTM 隐藏单元的权值
        '''
        return (torch.zeros((2, batch_size, self.hidden_size), requires_grad=True),
                torch.zeros((2, batch_size, self.hidden_size), requires_grad=True))
        # weight = next(self.parameters()).data
        # return (torch.FloatTensor(weight.new(2, batch_size, self.hidden_size).fill_(0)),
        #         torch.FloatTensor(weight.new(2, batch_size, self.hidden_size).fill_(0)))

    def sortTensor(self, padded_tensor, sequence_length):
        '''
        在pack之前，根据sequence_lens，将padded_tensor按降序排序
        '''
        sequence_length = torch.IntTensor(sequence_length)
        sequence_length, order = torch.sort(sequence_length, descending=True)
        padded_tensor_sorted = padded_tensor[order]
        # sequence_length = np.array(sequence_length, dtype="int32")
        # order = np.argsort(-padded_tensor)
        # padded_tensor = padded_tensor[order]
        # sequence_length = sequence_length[order]
        return padded_tensor_sorted, sequence_length, order

    def forward(self, s1, s2, s1_length, s2_length, s1_hidden, s2_hidden):
        batch_size = len(s1)

        s1_sorted, s1_length, s1_order = self.sortTensor(s1, s1_length)
        s2_sorted, s2_length, s2_order = self.sortTensor(s2, s2_length)

        s1_embeded = self.embedding(s1_sorted).float()
        s2_embeded = self.embedding(s2_sorted).float()

        s1_packed = pack_padded_sequence(s1_embeded, s1_length, batch_first=True)
        s2_packed = pack_padded_sequence(s2_embeded, s2_length, batch_first=True)

        s1_output, s1_hidden = self.lstm(s1_packed, s1_hidden)
        s2_output, s2_hidden = self.lstm(s2_packed, s2_hidden)

        s1_output, _ = pad_packed_sequence(s1_output, batch_first=True)
        s2_output, _ = pad_packed_sequence(s2_output, batch_first=True)

        # 恢复序列的原始顺序, 即排序前的顺序
        s1_output = s1_output[s1_order, :, :]
        s2_output = s2_output[s2_order, :, :]

        s1_representation = torch.zeros(s1_output.size(0), s1_output.size(2))
        s2_representation = torch.zeros(s2_output.size(0), s2_output.size(2))

        # 计算注意力权重大小，并加权求和
        for sent_idx in range(s1_output.size(0)):
            H = s1_output[sent_idx, :s1_length[sent_idx], : ]
            h1 = self.S1(H)
            h1 = F.tanh(h1)
            h2 = self.S2(h1)
            attention_weight = F.softmax(h2)
            # multiply the word vectors by their calculated attention weight
            weighted_sum = torch.mm(attention_weight.t(), H)
            s1_representation[sent_idx, :] = weighted_sum

        for sent_idx in range(s2_output.size(0)):
            H = s2_output[sent_idx, :s2_length[sent_idx], : ]
            h1 = self.S1(H)
            h1 = F.tanh(h1)
            h2 = self.S2(h1)
            attention_weight = F.softmax(h2)
            # multiply the word vectors by their calculated attention weight
            weighted_sum = torch.mm(attention_weight.t(), H)
            s2_representation[sent_idx, :] = weighted_sum

        merge_add = torch.add(s1_representation, s2_representation)
        neg_s2 = torch.neg(s2_representation)
        merge_minus = torch.add(s1_representation, neg_s2)
        merge_minus = torch.pow(merge_minus, 2)
        merged = torch.cat((merge_add, merge_minus), 1)
        merged = self.dropout(merged)

        output_mlp = self.mlp(merged)
        # output_mlp = self.dropout(output_mlp)

        output = self.output(output_mlp)
        output = self.sigmoid(output)
        return output


























































