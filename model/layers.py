import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_filter import Filter
from torch_sparse import spmm
import pygsp
from math import sqrt
import numpy as np


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

    def __init__(self, G, in_features, out_features, dropout, alpha, device,
                 out_channels, chebyshev_order=16, pre_training=False, filter="analysis"):
        super(GraphSpectralFilterLayer, self).__init__()
        self.G = G
        self.device=device
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.out_channels = out_channels
        self.alpha = alpha
        self.pre_training = pre_training
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.chebyshev_order = chebyshev_order
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.N = G.n_vertices
        self.filter_type = filter
        self.filter_kernel = AnalysisFilter(out_channel=self.out_channels) if self.filter_type == 'analysis' else GaussFilter(k=self.out_channels)
        self.filter = Filter(self.G, self.filter_kernel, chebyshev_order=self.chebyshev_order)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.filter_kernel = AnalysisFilter(out_channel=self.out_channels) if self.filter_type == 'analysis' else GaussFilter(k=self.out_channels)
        self.filter = Filter(self.G, self.filter_kernel, chebyshev_order=self.chebyshev_order)

        if self.pre_training:
            itersine = pygsp.filters.Itersine(self.G, self.out_channels)
            k_optimizer = torch.optim.Adam(self.filter_kernel.parameters(), lr=5e-4, weight_decay=1e-5)
            x = torch.rand(1000).view(-1, 1) * 2
            y = torch.FloatTensor(itersine.evaluate(x.cpu().view(-1))).T.to(self.device)
            val_x = torch.rand(1000).view(-1, 1) * 2
            val_y = torch.FloatTensor(itersine.evaluate(val_x.cpu().view(-1))).T.to(self.device)
            for _ in range(10000):
                self.filter_kernel.train()
                k_optimizer.zero_grad()
                predictions = self.filter_kernel(x)
                loss = F.mse_loss(input=predictions, target=y, reduction="mean")
                if _ % 1000 == 0:
                    self.filter_kernel.eval()
                    val_predictions = self.filter_kernel(val_x)
                    val_loss = F.mse_loss(input=val_predictions, target=val_y, reduction="mean")
                    # print(
                    #     'kernel training epoch {} loss {} validation loss {}'.format(_, str(loss.item()),
                    #                                                                  str(val_loss.item())))
                    self.filter_kernel.train()
                loss.backward()
                k_optimizer.step()

    def forward(self, input):
        h = torch.mm(input, self.W)
        N = h.shape[0]
        assert not torch.isnan(h).any()

        coefficients_list = self.filter()
        h_primes = []
        attentions = []
        for coefficients in coefficients_list:
            attention_indices = coefficients.indices()
            attention_values = coefficients.values()
            attention_values = self.leakyrelu(attention_values)
            attention_values = torch.where(torch.isnan(attention_values).logical_or(attention_values.eq(0)), torch.full_like(attention_values, -9e15), attention_values)
            attention_values = torch.exp(attention_values).clamp(max=9e15)
            divisor = spmm(attention_indices,
                           attention_values,
                           self.N,
                           self.N,
                           torch.ones(self.N, 1))
            # Avoid dividing by zero
            divisor = divisor.masked_fill(divisor == 0, 1)

            h_prime = spmm(attention_indices,
                           F.dropout(attention_values,
                                     training=self.training,
                                     p=self.dropout * attention_values.shape[0] / (self.N * self.N)),
                           self.N,
                           self.N,
                           h).div(divisor)
            assert not torch.isnan(h_prime).any()
            h_primes.append(F.elu(h_prime))
            attentions.append(torch.sparse_coo_tensor(attention_indices, attention_values, (N, N)).to_dense().div(divisor))
        return torch.cat(h_primes, dim=1), attentions

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


