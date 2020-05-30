import torch.nn as nn
import argparse
import torch.nn.functional as F
from spectral import GraphSpectralFilterLayer

from citation import get_planetoid_dataset, random_planetoid_splits, run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--heads', type=int, default=18)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--mlp-hidden', type=int, default=128)
parser.add_argument('--mlp-dropout', type=float, default=0.5)
parser.add_argument('--attention-dropout', type=float, default=0.8)
parser.add_argument('--feature-dropout', type=float, default=0.8)
parser.add_argument('--spectrum-dropout', type=float, default=0.8)
parser.add_argument('--chebyshev-order', type=float, default=16)
parser.add_argument('--pre_training', action='store_true', default=False, help='Initialize filters with Itersine')
args = parser.parse_args()


class Net(nn.Module):
    def __init__(self, dataset, mlp_hidden, mlp_dropout, kernel=None):
        super(Net, self).__init__()
        self.G = G
        self.G.estimate_lmax()
        self.feature_dropout = nn.Dropout(p=args.feature_dropout)
        self.leakyRelu = nn.LeakyReLU(args.alpha)

        self.analysis = GraphSpectralFilterLayer(self.G, dataset.num_features, args.hidden, dropout=args.attention_dropout,
                                                 out_channels=args.heads,
                                                 spectrum_dropout=args.spectrum_dropout, alpha=args.alpha,
                                                 kernel=kernel, chebyshev_order=args.chebyshev_order)

        self.synthesis = GraphSpectralFilterLayer(self.G, args.hidden * args.heads, args.hidden, dropout=args.attention_dropout,
                                                  out_channels=1,
                                                  spectrum_dropout=args.spectrum_dropout, alpha=args.alpha,
                                                  kernel=None, chebyshev_order=args.chebyshev_order)

        self.out_att = nn.Sequential(
            nn.Linear(args.hidden, mlp_hidden),
            nn.Dropout(p=mlp_dropout),
            nn.LeakyReLU(),
            nn.Linear(mlp_hidden, dataset.num_classes),
            nn.Dropout(p=mlp_dropout),
            nn.LeakyReLU(),
            # nn.Tanh()
        )

    def forward(self, data):
        x = data.x
        x = self.feature_dropout(x)
        x = self.leakyRelu(self.analysis(x))
        x = self.feature_dropout(x)
        x = self.synthesis(x)
        x = self.feature_dropout(x)
        x = self.leakyRelu(self.out_att(x))
        return F.log_softmax(x, dim=1)

dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks)
