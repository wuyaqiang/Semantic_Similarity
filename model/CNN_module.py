#coding=utf-8
from __future__ import print_function
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, embedding_dim, kernel_num, kernel_sizes, dropout_size):
        super(CNNModel, self).__init__()

        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.channel = 1

        # CNN layer
        self.conv1 = nn.ModuleList([nn.Conv2d(self.channel, kernel_num, (size, embedding_dim)) for size in kernel_sizes])
        self.conv2 = nn.ModuleList([nn.Conv2d(self.channel, kernel_num, (size, embedding_dim)) for size in kernel_sizes])
        # dropout layer
        self.dropout = nn.Dropout(dropout_size)
        # non linearity function
        self.relu = nn.ReLU()

    def forward(self, s1_embeded, s2_embeded):
        s1_embeded = s1_embeded.unsqueeze(1)    # (bs, channel, len, dim)
        s2_embeded = s2_embeded.unsqueeze(1)    # (bs, channel, len, dim)

        s1 = [self.relu(conv(s1_embeded)).squeeze(3) for conv in self.conv1]
        s2 = [self.relu(conv(s2_embeded)).squeeze(3) for conv in self.conv2]
        s1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in s1]
        s2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in s2]

        s1_representation = torch.cat(s1, 1)
        s2_representation = torch.cat(s2, 1)

        return s1_representation, s2_representation

