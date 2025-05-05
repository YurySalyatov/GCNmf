"""
version 1.0
date 2021/02/04
"""

import torch.nn as nn
import torch.nn.functional as F

from . import GCNConv, GCNmfConv


class GCNmf(nn.Module):
    def __init__(self, data, nhid=16, dropout=0.5, n_components=5):
        super(GCNmf, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.gc1 = GCNmfConv(nfeat, nhid, data, n_components, dropout)
        self.gc2 = GCNConv(nhid, nclass, dropout)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def set_dropout(self, dropout):
        self.dropout = dropout
        self.gc1.dropout = dropout
        self.gc2.dropout = dropout

    def set_eval(self):
        self.gc1.eval()
        self.gc2.eval()

    def set_train(self):
        self.gc1.train()
        self.gc2.train()

    def forward(self, data):
        x, adj = data.features, data.adj
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
