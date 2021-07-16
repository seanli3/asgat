from model.layers import GraphSpectralFilterLayer
from model.spectral_filter import Graph
import torch
import torch.nn.functional as F
from random import seed as rseed
from numpy.random import seed as nseed
from citation import get_dataset, run
from ax import optimize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--trials', type=int, default=40)
parser.add_argument('--split', type=str, default='full')
parser.add_argument('--lcc', type=bool, default=False)
parser.add_argument('--method', type=str, default='chebyshev')
parser.add_argument('--filter', type=str, default='analysis')
arg = parser.parse_args()

args = {
    'dataset': arg.dataset,
    'runs': 10,
    'cuda': True,
    'method': arg.method,
    'lcc': arg.lcc
}


def decimation(args):
    print(args)
    rseed(args['seed'])
    nseed(args['seed'])
    torch.manual_seed(args['seed'])

    args['cuda'] = args['cuda'] and torch.cuda.is_available()


    # if args['cuda']:
    #     print("-----------------------Training on CUDA-------------------------")

    if args['cuda']:
        torch.cuda.manual_seed(args['seed'])


    class Net(torch.nn.Module):
        def __init__(self, dataset):
            super(Net, self).__init__()
            data = dataset[0]
            if args['cuda']:
                data.to('cuda')
            self.G = Graph(data, lap_type='combinatorial' if args['method'] == 'lanzcos' else 'normalized')

            self.analysis = GraphSpectralFilterLayer(self.G, dataset.num_node_features, args['hidden'], method=args['method'],
                                                     dropout=args['dropout'], out_channels=args['heads'], filter=args['filter'],
                                                     pre_training=args['pre_training'], device='cuda' if args['cuda'] else 'cpu',
                                                     order=args['order'], concat=True, k=args['k'],
                                                     threshold=None,
                                                     Kb=args['Kb'], Ka=args['Ka'], Tmax=args['Tmax'], tau=args['tau'])
            # self.mlp = nn.Sequential(nn.Linear(args['hidden * args['heads, 128),
            #                             nn.ReLU(inplace=True),
            #                             nn.Linear(128, 64),
            #                             nn.ReLU(inplace=True),
            #                             nn.Linear(64, 32),
            #                             nn.ReLU(inplace=True),
            #                             nn.Linear(32, dataset.num_classes),
            #                             nn.ReLU(inplace=True))

            # self.W = torch.zeros(args['hidden * args['heads, dataset.num_classes)
            self.synthesis = GraphSpectralFilterLayer(self.G, args['hidden'] * args['heads'], dataset.num_classes, method=args['method'],
                                     dropout=args['dropout'], out_channels=1, filter=args['filter'],
                                     pre_training=args['pre_training'], device='cuda' if args['cuda'] else 'cpu',
                                     order=args['order'], concat=True, k=args['k'],
                                     threshold=None,
                                     Kb=args['Kb'], Ka=args['Ka'], Tmax=args['Tmax'], tau=args['tau'])

            if args['cuda']:
                self.to('cuda')

        def reset_parameters(self):
            self.analysis.reset_parameters()
            # torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
            # for layer in self.mlp:
            #     if hasattr(layer, 'reset_parameters'):
            #         layer.reset_parameters()
            self.synthesis.reset_parameters()
            if args['cuda']:
                self.to('cuda')

        def forward(self, data):
            x = data.x
            x = F.dropout(x, p=args['dropout'], training=self.training)
            x, att1 = self.analysis(x)
            x = F.dropout(x, p=args['dropout'], training=self.training)
            x, att2 = self.synthesis(x)
            x = F.elu(x)
            # x = F.elu(x.mm(self.W))
            # x = self.mlp(x)
            return F.log_softmax(x, dim=1), att1, att2

    use_dataset = lambda: get_dataset(args['dataset'], args['normalize_features'], edge_dropout=args['edge_dropout'],
                                      permute_masks=None, cuda=args['cuda'], split=args['split'],lcc=args['lcc'],
                                      node_feature_dropout=args['node_feature_dropout'], self_loop=args['self_loop'])

    return run(use_dataset, Net, args['runs'], args['epochs'], args['lr'], args['weight_decay'],
        args['patience'], None, cuda=args['cuda'])


parameters=[
    {'name': 'dataset', 'type': 'fixed', 'value': args['dataset']},
    {'name': 'runs', 'type': 'fixed', 'value': 1},
    {'name': 'epochs', 'type': 'fixed', 'value': 1000},
    {'name': 'seed', 'type': 'fixed', 'value': 729},
    {'name': 'lr', 'type': 'range', "type": "choice", "values": [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1]},
    {'name': 'weight_decay', 'type': 'choice', "values": [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]},
    {'name': 'patience', 'type': 'fixed', 'value': 100},
    {'name': 'hidden', 'type': 'choice', "values": [16, 32, 64, 88, 128, 256, 512]},
    {'name': 'heads', 'type': 'range', "bounds": [1, 18]},
    {'name': 'dropout', "type": "choice", "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    {'name': 'self_loop', "type": "choice", "values": [False ,True]},
    {'name': 'normalize_features', "type": "fixed", "value": True},
    {'name': 'pre_training', "type": "fixed", "value": False},
    {'name': 'cuda', 'type': 'fixed', 'value': args['cuda']},
    {'name': 'split', 'type': 'fixed', 'value': arg.split},
    {'name': 'lcc', 'type': 'fixed', 'value': arg.lcc},
    {'name': 'method', 'type': 'fixed', 'value': arg.method},
    {'name': 'edge_dropout', 'type': 'fixed', 'value': 0},
    {'name': 'node_feature_dropout', 'type': 'fixed', 'value': 0},
    {'name': 'filter', 'type': 'fixed', 'value': arg.filter},
    {'name': 'k', 'type': 'range', 'bounds': [2, 18]}
]

if arg.method.lower() == 'chebyshev' or arg.method.lower() == 'lanzcos':
    parameters += [
        {'name': 'order', 'type': 'choice', "values": [8, 10, 12, 14, 16, 18]},
        {'name': 'Kb', 'type': 'fixed', "value": 1},
        {'name': 'Ka', 'type': 'fixed', "value": 1},
        {'name': 'Tmax', 'type': 'fixed', "value": 20},
    ]
elif arg.method.lower() == 'arma':
    parameters += [
        {'name': 'order', 'type': 'fixed', "value": 16},
        {'name': 'Kb', 'type': 'range', "bounds": [1, 20]},
        {'name': 'Ka', 'type': 'range', "bounds": [1, 20]},
        {'name': 'Tmax', 'type': 'range', "bounds": [50, 2000], "log_scale": True},
    ]


if arg.filter == 'heat':
    parameters += [
        {'name': 'tau', 'type': 'range', 'bounds': [0., 1.]}
    ]
else:
    parameters += [
        {'name': 'tau', 'type': 'fixed', 'value': 0.2}
    ]

best_parameters, best_values, _, _ = optimize(
    parameters=parameters,
    evaluation_function=decimation,
    total_trials=arg.trials,
    minimize=False)

print(best_parameters, best_values)
