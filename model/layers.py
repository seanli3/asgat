import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_filter import Filter
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
    def __init__(self, out_channel, device):
        super(AnalysisFilter, self).__init__()
        self.device=device
        self.out_channel = out_channel
        self.layers = nn.Sequential(nn.Linear(1, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 64),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(64, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, out_channel),
                                    nn.ReLU(inplace=True))
        self.to(device)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.to(self.device)

    def forward(self, x):
        return self.layers(x.view(-1,1))


class GraphSpectralFilterLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, G, in_features, out_features, dropout, alpha, out_channels, device="cpu", concat=True,
                 chebyshev_order=16, pre_training=False, filter="analysis"):
        super(GraphSpectralFilterLayer, self).__init__()
        self.G = G
        self.device=device
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.out_channels = out_channels
        self.alpha = alpha
        self.pre_training = pre_training
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # self.mlp = nn.Sequential(
        #     nn.Linear(in_features, 512, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(512, out_features, bias=False),
        # )
        self.chebyshev_order = chebyshev_order
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.N = G.n_vertices
        self.filter_type = filter
        self.filter_kernel = AnalysisFilter(out_channel=self.out_channels, device=self.device) if self.filter_type == 'analysis' else GaussFilter(k=self.out_channels)
        self.filter = Filter(self.G, self.filter_kernel, nf=self.out_channels, device=self.device, chebyshev_order=self.chebyshev_order)
        self.concat = concat

        self.to(self.device)
        self.linear.to(self.device)
        self.filter_kernel.to(self.device)

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.linear.reset_parameters()
        # for layer in self.mlp:
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()
        self.filter_kernel = AnalysisFilter(out_channel=self.out_channels, device=self.device) if self.filter_type == 'analysis' else GaussFilter(k=self.out_channels)
        self.filter = Filter(self.G, self.filter_kernel, nf=self.out_channels, device=self.device, chebyshev_order=self.chebyshev_order)

        if self.pre_training:
            itersine = pygsp.filters.Itersine(self.G, self.out_channels)
            k_optimizer = torch.optim.Adam(self.filter_kernel.parameters(), lr=5e-4, weight_decay=1e-5)
            x = torch.rand(1000, device=self.device).view(-1, 1) * 2
            y = torch.FloatTensor(itersine.evaluate(x.cpu().view(-1))).to(self.device).T
            val_x = torch.rand(1000, device=self.device).view(-1, 1) * 2
            val_y = torch.FloatTensor(itersine.evaluate(val_x.cpu().view(-1))).to(self.device).T
            for _ in range(10000):
                self.filter_kernel.train()
                k_optimizer.zero_grad()
                predictions = self.filter_kernel(x)
                loss = F.mse_loss(input=predictions, target=y, reduction="mean")
                if _ % 1000 == 0:
                    self.filter_kernel.eval()
                    val_predictions = self.filter_kernel(val_x)
                    val_loss = F.mse_loss(input=val_predictions, target=val_y, reduction="mean")
                    print(
                        'kernel training epoch {} loss {} validation loss {}'.format(_, str(loss.item()),
                                                                                     str(val_loss.item())))
                    self.filter_kernel.train()
                loss.backward()
                k_optimizer.step()
        self.to(self.device)
        self.linear.to(self.device)
        self.filter_kernel.to(self.device)

    def forward(self, input):
        # h = torch.mm(input, self.W)
        h = self.linear(input)
        N = h.shape[0]
        assert not torch.isnan(h).any()

        attention = self.filter()
        attentions = []

        overall_mean = attention.mean()
        attention = torch.where(attention > overall_mean, attention, torch.tensor([-9e15], device=self.device))
        attention = self.leakyrelu(attention)
        attention = attention.softmax(0)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = attention.mm(h)
        assert not torch.isnan(h_prime).any()

        if self.concat:
            return h_prime.view(self.out_channels, N, self.out_features).permute(1, 0, 2).reshape(N, -1), attentions
        else:
            return h_prime.view(self.out_channels, N, self.out_features).mean(dim=0), attentions

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


