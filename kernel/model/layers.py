import torch
from .spectral_filter import Graph
import torch.nn as nn
import torch.nn.functional as F
from .spectral_filter import Filter
from torch_sparse import spmm
import pygsp
from math import sqrt


class GaussFilter(nn.Module):
    def __init__(self, k: int):
        super(GaussFilter, self).__init__()
        self.k = k
        self.reset_parameters()
        self.centers = torch.linspace(0, 2, k)

    def reset_parameters(self):
        self.bandwidths = nn.Parameter(torch.full([1, self.k], sqrt(2/self.k)))
        self.gains = nn.Parameter(torch.ones(1, self.k))
        self.amps = nn.Parameter(torch.full([1, self.k], 8.0))

    def forward(self, x):
        return self.gains.mul(torch.exp(-self.amps*(x.view(-1, 1) - self.centers).pow(2)/self.bandwidths.pow(2)))


class AnalysisFilter(nn.Module):
    def __init__(self, out_channel):
        super(AnalysisFilter, self).__init__()
        self.out_channel = out_channel
        self.layers = nn.Sequential(nn.Linear(1, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 64),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(64, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, out_channel),
                                    nn.ReLU(inplace=True))

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        return self.layers(x.view(-1,1))


class GraphSpectralFilterLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, out_channels, device="cpu", concat=True,
                 order=16, pre_training=False, filter="analysis"):
        super(GraphSpectralFilterLayer, self).__init__()
        self.device=device
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.out_channels = out_channels
        self.alpha = alpha
        self.pre_training = pre_training
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.order = order
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.filter_type = filter
        self.filter_kernel = AnalysisFilter(out_channel=self.out_channels) if self.filter_type == 'analysis' else GaussFilter(k=self.out_channels)
        self.filter = Filter(self.filter_kernel, order=self.order)
        self.concat = concat

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.filter_kernel = AnalysisFilter(out_channel=self.out_channels) if self.filter_type == 'analysis' else GaussFilter(k=self.out_channels)
        self.filter = Filter(self.filter_kernel, order=self.order)

    def forward(self, input, edge_index):
        adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]))
        G = Graph(adj)
        G.estimate_lmax()

        h = torch.mm(input, self.W)
        N = h.shape[0]
        assert not torch.isnan(h).any()

        coefficients_list = self.filter(G)
        h_primes = []
        attentions = []

        for coefficients in coefficients_list:
            # overall_mean = torch.sparse.sum(coefficients) / N / N
            attention_indices = coefficients.indices()
            attention_values = coefficients.values()
            attention_values = self.leakyrelu(attention_values)
            attention_values = torch.where(torch.isnan(attention_values).logical_or(attention_values.lt(0)), torch.full_like(attention_values, -9e15), attention_values)
            attention_values = torch.exp(attention_values).clamp(max=9e15)
            divisor = spmm(attention_indices,
                           attention_values,
                           N,
                           N,
                           torch.ones(N, 1))
            # Avoid dividing by zero
            divisor = divisor.masked_fill(divisor == 0, 1)

            h_prime = spmm(attention_indices,
                           F.dropout(attention_values,
                                     training=self.training,
                                     p=self.dropout * attention_values.shape[0] / (N * N)),
                           N,
                           N,
                           h).div(divisor)
            assert not torch.isnan(h_prime).any()
            h_primes.append(F.elu(h_prime))
            # attentions.append(torch.sparse_coo_tensor(attention_indices, attention_values, (N, N)).to_dense().div(divisor))

        if self.concat:
            return torch.cat(h_primes, dim=1)
        else:
            return torch.stack(h_primes, dim=2).mean(dim=2)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


