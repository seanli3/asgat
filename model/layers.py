import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_filter import Filter
from torch_sparse import spmm


class GraphSpectralFilterLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, G, in_features, out_features, dropout, out_channels, device="cpu", concat=True,
                 order=16, pre_training=False, filter="analysis", method="chebyshev", k=5, threshold=None,
                 Kb=18, Ka=2, Tmax=200):
        super(GraphSpectralFilterLayer, self).__init__()
        self.G = G
        self.k = k
        self.threshold = threshold
        self.Kb = Kb
        self.Ka = Ka
        self.Tmax = Tmax
        self.device=device
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.out_channels = out_channels
        self.method = method
        self.pre_training = pre_training
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # self.mlp = nn.Sequential(
        #     nn.Linear(in_features, 512, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(512, out_features, bias=False),
        # )
        self.order = order
        self.N = G.n_vertices
        self.filter_type = filter
        self.filter = Filter(self.G, nf=self.out_channels, device=self.device, order=self.order,
                             method=self.method, Kb=self.Kb, Ka=self.Ka, Tmax=self.Tmax)
        self.concat = concat

        self.to(self.device)
        self.linear.to(self.device)

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.linear.reset_parameters()
        # for layer in self.mlp:
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()
        self.filter = Filter(self.G, nf=self.out_channels, device=self.device, order=self.order,
                             method=self.method, Kb=self.Kb, Ka=self.Ka, Tmax=self.Tmax)
        self.filter.reset_parameters()

        self.to(self.device)
        self.linear.to(self.device)

    def forward(self, input):
        h = self.linear(input)
        assert not torch.isnan(h).any()

        attention = self.filter()

        if self.threshold is not None:
            attention = torch.where(attention > self.threshold, attention, torch.tensor([-9e15], device=self.device))
        else:
            ret = torch.topk(attention, k=self.k, dim=1)
            attention.fill_(-9e15)
            attention.scatter_(1, ret.indices, ret.values)

        attention = attention.softmax(1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        nzidx = attention.nonzero(as_tuple=True)
        h_prime = spmm(nzidx, attention[nzidx], self.N*self.out_channels, self.N, h)

        if self.concat:
            return h_prime.view(self.out_channels, self.N, self.out_features).permute(1, 0, 2).reshape(self.N, -1), attention
        else:
            return h_prime.view(self.out_channels, self.N, self.out_features).mean(dim=0), attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


