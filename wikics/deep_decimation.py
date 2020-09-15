import argparse
from model.layers import GraphSpectralFilterLayer, AnalysisFilter
from model.spectral_filter import Graph
import torch
import torch.nn.functional as F
from random import seed as rseed
from numpy.random import seed as nseed
from citation import get_planetoid_dataset, random_planetoid_splits, run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--alpha', type=float, default=0.4325176513738335)
parser.add_argument('--seed', type=int, default=729, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.0001502755734819816)
parser.add_argument('--weight_decay', type=float, default=1e-04)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--hidden', type=int, default=83)
parser.add_argument('--heads', type=int, default=19)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--pre_training', action='store_true')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--chebyshev_order', type=int, default=19, help='Chebyshev polynomial order')
parser.add_argument('--edge_dropout', type=float, default=0)
parser.add_argument('--node_feature_dropout', type=float, default=0)
parser.add_argument('--filter', type=str, default='analysis')
parser.add_argument('--layers', type=int, default=1)
args = parser.parse_args()

print(args)

rseed(args.seed)
nseed(args.seed)
torch.manual_seed(args.seed)

args.cuda = args.cuda and torch.cuda.is_available()


if args.cuda:
    print("-----------------------Training on CUDA-------------------------")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        data = dataset.data
        adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.num_edges))
        self.G = Graph(adj)
        self.G.estimate_lmax()

        self.analysis = GraphSpectralFilterLayer(self.G, dataset.num_node_features, args.hidden,
                                                 dropout=args.dropout, out_channels=args.heads,
                                                 pre_training=args.pre_training, device='cuda' if args.cuda else 'cpu',
                                                 alpha=args.alpha, chebyshev_order=args.chebyshev_order)

        self.synthesis = GraphSpectralFilterLayer(self.G, args.hidden * args.heads, dataset.num_classes,
                                                  device='cuda' if args.cuda else 'cpu', dropout=args.dropout,
                                                  out_channels=1, alpha=args.alpha, pre_training=False,
                                                  chebyshev_order=args.chebyshev_order)

        self.layers = []
        for _ in range(args.layers):
            self.layers.append(GraphSpectralFilterLayer(self.G, args.hidden * args.heads, args.hidden,
                                                        device='cuda' if args.cuda else 'cpu', dropout=args.dropout,
                                                        out_channels=args.heads, alpha=args.alpha,
                                                        pre_training=args.pre_training,
                                                        chebyshev_order=args.chebyshev_order))

    def reset_parameters(self):
        self.analysis.reset_parameters()
        self.synthesis.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.analysis(x)[0]
        for layer in self.layers:
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = layer(x)[0]
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = F.elu(self.synthesis(x)[0])
        return F.log_softmax(x, dim=1), None, None


dataset = get_planetoid_dataset(args.dataset, args.normalize_features, edge_dropout=args.edge_dropout,
                                node_feature_dropout=args.node_feature_dropout)
if args.cuda:
    dataset.data.to('cuda')

permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.patience, permute_masks)
