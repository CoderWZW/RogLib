import numpy as np
import scipy.sparse as sp
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, input_features, adj):
        support = torch.mm(input_features, self.weight)
        output = torch.spmm(adj, support)
        if self.use_bias:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, sizes, input_dim):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, sizes[0])
        self.gcn2 = GraphConvolution(sizes[0], sizes[1])
        pass
    
    def forward(self, X, adj):
        X = F.relu(self.gcn1(X, adj))
        X = self.gcn2(X, adj)
        
        return F.log_softmax(X, dim=1)