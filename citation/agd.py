import torch.nn as nn
import argparse
import torch.nn.functional as F
from torch import optim
from spectral import GraphSpectralFilterLayer, Graph, AnalysisFilter
import torch
import pygsp

from citation import get_planetoid_dataset, random_planetoid_splits, run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--heads', type=int, default=18)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--mlp-hidden', type=int, default=128)
parser.add_argument('--mlp-dropout', type=float, default=0.5)
parser.add_argument('--attention-dropout', type=float, default=0.9)
parser.add_argument('--feature-dropout', type=float, default=0.8)
parser.add_argument('--spectrum-dropout', type=float, default=0)
parser.add_argument('--chebyshev-order', type=float, default=16)
parser.add_argument('--pre_training', action='store_true', default=False, help='Initialize filters with Itersine')
args = parser.parse_args()


class Net(nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        N = dataset.data.num_nodes
        adj = torch.sparse_coo_tensor(dataset.data.edge_index, torch.ones(dataset.data.num_edges), [N, N])
        self.G = Graph(adj)
        self.G.estimate_lmax()
        self.feature_dropout = nn.Dropout(p=args.feature_dropout)
        self.elu = nn.ELU(args.alpha)
        self.kernel = None

        self.analysis = GraphSpectralFilterLayer(self.G, dataset.num_features, args.hidden, dropout=args.attention_dropout,
                                                 out_channels=args.heads,
                                                 spectrum_dropout=args.spectrum_dropout,
                                                 kernel=self.kernel, chebyshev_order=args.chebyshev_order)

        self.synthesis = GraphSpectralFilterLayer(self.G, args.hidden * args.heads, dataset.num_classes, dropout=args.attention_dropout,
                                                  out_channels=1,
                                                  spectrum_dropout=args.spectrum_dropout,
                                                  kernel=None, chebyshev_order=args.chebyshev_order)

    def reset_parameters(self):
        self.analysis.reset_parameters()
        self.synthesis.reset_parameters()
        if args.pre_training:
            itersine = pygsp.filters.Itersine(self.G, args.heads)
            self.kernel = AnalysisFilter(args.heads, args.spectrum_dropout)
            k_optimizer = optim.Adam(self.kernel.parameters(), lr=5e-4, weight_decay=1e-5)
            x = torch.rand(1000).view(-1, 1) * 2
            y = torch.FloatTensor(itersine.evaluate(x.cpu().view(-1))).T
            val_x = torch.rand(1000).view(-1, 1) * 2
            val_y = torch.FloatTensor(itersine.evaluate(val_x.cpu().view(-1))).T
            for _ in range(8000):
                self.kernel.train()
                k_optimizer.zero_grad()
                predictions = self.kernel(x)
                loss = F.mse_loss(input=predictions, target=y, reduction="mean")
                if _ % 1000 == 0:
                    self.kernel.eval()
                    val_predictions = self.kernel(val_x)
                    val_loss = F.mse_loss(input=val_predictions, target=val_y, reduction="mean")
                    print(
                        'kernel training epoch {} loss {} validation loss {}'.format(_, str(loss.item()),
                                                                                     str(val_loss.item())))
                    self.kernel.train()
                loss.backward()
                k_optimizer.step()

    def forward(self, data):
        x = data.x
        x = self.feature_dropout(x)
        x = self.elu(self.analysis(x))
        x = self.feature_dropout(x)
        x = self.elu(self.synthesis(x))
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks)
