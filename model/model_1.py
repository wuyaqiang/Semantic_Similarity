#coding=utf-8
from __future__ import print_function

import numpy as np
import torch
from torch import nn
from .CNN_module import CNNModel

class MainModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, context_vec_size, mlp_hid_size, kernel_num, kernel_sizes,
                 embedding_dict, word_dict, dropout_size, class_size, use_cuda):
        super(MainModel, self).__init__()

        self.use_cuda = use_cuda
        self.word_dict = word_dict
        self.embedding_dict = embedding_dict
        self.hidden_size = hidden_size
        self.context_vec_size = context_vec_size
        self.mlp_hid_size = mlp_hid_size

        # self-attention layer
        self.S1 = nn.Linear(hidden_size * 2, context_vec_size, bias=False)
        self.S2 = nn.Linear(context_vec_size, 1, bias=False)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # bi-LSTM layer
        self.lstm1 = nn.LSTM(embedding_dim, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(embedding_dim, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        # CNN layer
        self.cnn_dim = len(kernel_sizes) * kernel_num
        self.cnn = CNNModel(embedding_dim, kernel_num, kernel_sizes, dropout_size)
        # mlp layer
        self.representation_dim = (hidden_size * 4 + self.cnn_dim) * 2
        self.mlp1 = nn.Linear(hidden_size * 4, mlp_hid_size)
        self.mlp2 = nn.Linear(hidden_size * 2, mlp_hid_size)
        # output layer
        self.output = nn.Linear(mlp_hid_size, class_size)
        # Similarity function
        self.CosineSim = nn.CosineSimilarity(dim=3)
        # normalization layer
        self.dropout = nn.Dropout(dropout_size)
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
        s1_embeded = self.embedding(s1)
        s2_embeded = self.embedding(s2)

        s1_cnn, s2_cnn = self.cnn(s1_embeded, s2_embeded)

        s1_output, s1_hidden = self.lstm1(s1_embeded, s1_hidden)
        s2_output, s2_hidden = self.lstm1(s2_embeded, s2_hidden)
        s1_output = self.layer_norm(s1_output)
        s2_output = self.layer_norm(s2_output)

        '''co-attention 权重计算方式'''
        s1_expanded = s1_output.unsqueeze(2).expand(s1_output.size()[0], s1_output.size()[1],
                                                    s2_output.size()[1], s1_output.size()[2])
        s2_expanded = s2_output.unsqueeze(1).expand(s2_output.size()[0], s1_output.size()[1],
                                                    s2_output.size()[1], s2_output.size()[2])

        cosine_sim = self.CosineSim(s1_expanded, s2_expanded)
        s1_attentive = torch.matmul(self.softmax(cosine_sim), s2_output)
        s2_attentive = torch.matmul(self.softmax(cosine_sim.transpose(1,2)), s1_output)
        s1_attentive = self.layer_norm(s1_attentive)
        s2_attentive = self.layer_norm(s2_attentive)

        # s1_concat = torch.cat((s1_output, s1_attentive), 2)
        # s2_concat = torch.cat((s2_output, s2_attentive), 2)

        # s1_representation, _ = torch.max(s1_concat, 1)
        # s2_representation, _ = torch.max(s2_concat, 1)

        # s1_representation_mean = torch.mean(s1_concat, 1, keepdim=False)
        # s2_representation_mean = torch.mean(s2_concat, 1, keepdim=False)

        #s1_representation = torch.cat((s1_representation, s1_cnn), 1)
        #s2_representation = torch.cat((s2_representation, s2_cnn), 1)
        #s1_representation = torch.cat((s1_representation_max, s1_representation_mean), 1)
        #s2_representation = torch.cat((s2_representation_max, s2_representation_mean), 1)

        '''计算s1的每个hidden和s2的hidden_mean之间的注意力权重'''
        # s2_hidden_mean = torch.mean(s2_output, 1, keepdim=True)
        # s2_hidden_mean = torch.transpose(s2_hidden_mean, 1, 2).contiguous()
        # s1_attention_weight = torch.matmul(s1_output, s2_hidden_mean)
        # s1_attention_weight = self.softmax(s1_attention_weight)
        # s1_attention_weight = torch.transpose(s1_attention_weight, 1, 2).contiguous()
        # s1_representation = torch.matmul(s1_attention_weight, s1_output).squeeze()
        '''计算s2的每个hidden和s1的hidden_mean之间的注意力权重'''
        # s1_hidden_mean = torch.mean(s1_output, 1, keepdim=True)
        # s1_hidden_mean = torch.transpose(s1_hidden_mean, 1, 2).contiguous()
        # s2_attention_weight = torch.matmul(s2_output, s1_hidden_mean)
        # s2_attention_weight = self.softmax(s2_attention_weight)
        # s2_attention_weight = torch.transpose(s2_attention_weight, 1, 2).contiguous()
        # s2_representation = torch.matmul(s2_attention_weight, s2_output).squeeze()

        '''计算s1的每个hidden和s2的最后一个hidden之间的注意力权重'''
        # s2_last_hidden = s2_hidden[0]
        # s2_last_hidden = torch.transpose(s2_last_hidden, 0, 1).contiguous()  # 注意: transpose之后一定要进行contiguous()
        # s2_last_hidden_size = s2_last_hidden.size()
        # s2_last_hidden = s2_last_hidden.view(s2_last_hidden_size[0], -1, 1)
        # s1_attention_weight = torch.matmul(s1_output, s2_last_hidden)
        # s1_attention_weight = self.softmax(s1_attention_weight)
        # s1_attention_weight = torch.transpose(s1_attention_weight, 1, 2).contiguous()
        # s1_representation = torch.matmul(s1_attention_weight, s1_output).squeeze()
        '''计算s2的每个hidden和s1的最后一个hidden之间的注意力权重'''
        # s1_last_hidden = s1_hidden[0]
        # s1_last_hidden = torch.transpose(s1_last_hidden, 0, 1).contiguous()
        # s1_last_hidden_size = s1_last_hidden.size()
        # s1_last_hidden = s1_last_hidden.view(s1_last_hidden_size[0], -1, 1)
        # s2_attention_weight = torch.matmul(s2_output, s1_last_hidden)
        # s2_attention_weight = self.softmax(s2_attention_weight)
        # s2_attention_weight = torch.transpose(s2_attention_weight, 1, 2).contiguous()
        # s2_representation = torch.matmul(s2_attention_weight, s2_output).squeeze()

        '''计算s1的self-attention'''
        # s1_H = s1_output
        # s1_h1 = self.S1(s1_H)
        # s1_h1 = self.tanh(s1_h1)
        # s1_h2 = self.S2(s1_h1)
        # attention_weight = self.softmax(s1_h2)
        # attention_weight = torch.transpose(attention_weight, 1, 2)
        # s1_representation = torch.matmul(attention_weight, s1_H).squeeze()
        '''计算s2的self-attention'''
        # s2_H = s2_output
        # s2_h1 = self.S1(s2_H)
        # s2_h1 = self.tanh(s2_h1)
        # s2_h2 = self.S2(s2_h1)
        # attention_weight = self.softmax(s2_h2)
        # attention_weight = torch.transpose(attention_weight, 1, 2)
        # s2_representation = torch.matmul(attention_weight, s2_H).squeeze()

        '''merged = [ s1 + s2 ; pow((s1 - s2), 2) ]'''
        merge_add = torch.add(s1_representation, s2_representation)
        neg_s2 = torch.neg(s2_representation)
        merge_minus = torch.add(s1_representation, neg_s2)
        merge_minus = torch.pow(merge_minus, 2)
        merged = torch.cat((merge_add, merge_minus), 1)

        '''merged = [ s1 ; s2 ; s1 + s2 ; s1 - s2 ; |s1 - s2| ]'''
        # merge_add = torch.add(s1_representation, s2_representation)
        # neg_s2 = torch.neg(s2_representation)
        # merge_minus = torch.add(s1_representation, neg_s2)
        # merge_abs = torch.abs(merge_minus)
        # merged = torch.cat((s1_representation, s2_representation, merge_add, merge_minus, merge_abs), 1)

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
        nn.init.xavier_normal_(self.S1.weight.data)
        nn.init.xavier_normal_(self.S2.weight.data)
        # self.S1.weight.data.uniform_(-init_range, init_range)
        # self.S2.weight.data.uniform_(-init_range, init_range)

        nn.init.xavier_normal_(self.mlp1.weight.data)
        nn.init.xavier_normal_(self.mlp2.weight.data)
        # self.mlp.weight.data.uniform_(-init_range, init_range)
        self.mlp1.bias.data.fill_(0)
        self.mlp2.bias.data.fill_(0)

        nn.init.xavier_normal_(self.output.weight.data)
        # self.output.weight.data.uniform_(-init_range, init_range)
        self.output.bias.data.fill_(0)

    def init_pretrained_embedding(self):
        # 初始化预训练的词向量矩阵
        nn.init.xavier_normal_(self.embedding.weight.data)
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
        hidden = torch.zeros((4, batch_size, self.hidden_size), requires_grad=True)
        cell = torch.zeros((4, batch_size, self.hidden_size), requires_grad=True)
        if self.use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return (hidden, cell)
        # weight = next(self.parameters()).data
        # return (torch.FloatTensor(weight.new(2, batch_size, self.hidden_size).fill_(0)),
        #         torch.FloatTensor(weight.new(2, batch_size, self.hidden_size).fill_(0)))

















