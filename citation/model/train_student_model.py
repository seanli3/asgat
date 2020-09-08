from model.layers import GraphSpectralFilterLayer, AnalysisFilter
import argparse
from model.spectral_filter import Graph
import torch
import torch.nn.functional as F
from torch import nn
from random import seed as rseed
from numpy.random import seed as nseed
from citation import get_dataset, random_planetoid_splits, run
from citation.train_eval import evaluate
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--hidden', type=int, default=88)
parser.add_argument('--heads', type=int, default=12)
parser.add_argument('--model_path', type=str, default='best_{}_gpu.pkl')
args = parser.parse_args()
print(args)

dataset_name = 'PubMed'

random_splits = False
runs = args.runs
epochs =args.epochs
alpha = 0.2
seed = 729
weight_decay = 0.00012376256876336363
patience = args.patience
hidden = args.hidden
lr = args.lr
dropout = 0.7
heads = args.heads
output_heads = 10
normalize_features = True
pre_training = False
cuda = True
chebyshev_order = 16
edge_dropout = 0
node_feature_dropout = 0
filter_name = 'analysis'

rseed(seed)
nseed(seed)
torch.manual_seed(seed)

cuda = cuda and torch.cuda.is_available()

if cuda:
    torch.cuda.manual_seed(seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


#%%

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        data = dataset.data
        adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.num_edges))
        self.G = Graph(adj)
        self.G.estimate_lmax()

        self.analysis = GraphSpectralFilterLayer(self.G, dataset.num_node_features, hidden,
                                                 dropout=dropout, out_channels=heads, filter=filter_name,
                                                 pre_training=False, device='cuda' if cuda else 'cpu',
                                                 alpha=alpha, chebyshev_order=chebyshev_order, concat=True)

        self.synthesis = GraphSpectralFilterLayer(self.G, hidden * heads, dataset.num_classes, filter=filter_name,
                                                  device='cuda' if cuda else 'cpu', dropout=dropout,
                                                  out_channels=1, alpha=alpha, pre_training=False,
                                                  chebyshev_order=chebyshev_order, concat=False)

    def reset_parameters(self):
        self.analysis.reset_parameters()
        self.synthesis.reset_parameters()

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=dropout, training=self.training)
        x, att1 = self.analysis(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x, att2 = self.synthesis(x)
        last_layer = x
        x = F.elu(x)
        return F.log_softmax(x, dim=1), last_layer, None

dataset = get_dataset(dataset_name, normalize_features, edge_dropout=edge_dropout,
                                node_feature_dropout=node_feature_dropout)

if cuda:
    dataset[0].to('cuda')


student_heads = 1


#%%

class StudentNet(torch.nn.Module):
    def __init__(self, dataset):
        super(StudentNet, self).__init__()
        data = dataset.data
        adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.num_edges))
        self.G = Graph(adj)
        self.G.estimate_lmax()

        self.analysis = GraphSpectralFilterLayer(self.G, dataset.num_node_features, dataset.num_classes,
                                                 dropout=dropout, out_channels=student_heads, filter=filter_name,
                                                 pre_training=False, device='cuda' if cuda else 'cpu',
                                                 alpha=alpha, chebyshev_order=chebyshev_order, concat=False)

        # self.synthesis = GraphSpectralFilterLayer(self.G, hidden * student_heads, dataset.num_classes, filter=filter_name,
        #                                           device='cuda' if cuda else 'cpu', dropout=dropout,
        #                                           out_channels=student_heads, alpha=alpha, pre_training=False,
        #                                           chebyshev_order=chebyshev_order, concat=False)
    def reset_parameters(self):
        self.analysis.reset_parameters()

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=dropout, training=self.training)
        x, att1 = self.analysis(x)
        # x = F.dropout(x, p=dropout, training=self.training)
        # last_layer, att2 = self.synthesis(x)
        last_layer = x
        x = F.elu(x)
        return F.log_softmax(x, dim=1), last_layer, None

#%%

model = Net(dataset)
model.load_state_dict(torch.load('./model/{}'.format(args.model_path),  map_location={'cuda:0': 'cpu'}))

# filter_kernel = model.analysis.filter_kernel

# model_correct_indices = get_correctly_predicted_node_idx(model, 'test', dataset)
eval_info = evaluate(model, dataset[0])
print('saved model', eval_info)

#%%

with torch.no_grad():
    soft_target = model(dataset[0])[1]

#%%

from torch.optim import Adam
from sklearn.metrics import f1_score


for _ in runs:
    print('Runs', _)
    student = StudentNet(dataset)
    student.reset_parameters()
    optimizer = Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    data = dataset.data
    dropout=0
    best_val_loss = float('inf')
    best_val_acc = float(0)
    eval_info_early_model = None
    bad_counter = 0

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits, out, _ = student(data)
        loss = F.mse_loss(out, soft_target)
        loss.backward()
        optimizer.step()
        student.train()

        # eval_info['epoch'] = epoch
        if epoch % 100 == 0:
            student.eval()
            outs = {}
            outs['loss'] = loss.item()
            outs['epoch'] = epoch
            for key in ['train', 'val', 'test']:
                mask = data['{}_mask'.format(key)]
                loss = F.nll_loss(logits[mask], data.y[mask]).item()
                pred = logits[mask].max(1)[1]
                outs['{}_loss'.format(key)] = loss
                outs['{}_micro_f1'.format(key)] = f1_score(data.y[mask].cpu(), logits[mask].max(1)[1].cpu(), average='micro')
                outs['{}_macro_f1'.format(key)] = f1_score(data.y[mask].cpu(), logits[mask].max(1)[1].cpu(), average='macro')

            print(outs)
        if eval_info['val_acc'] > best_val_acc or eval_info['val_loss'] < best_val_loss:
            if eval_info['val_acc'] >= best_val_acc and eval_info['val_loss'] <= best_val_loss:
                eval_info_early_model = eval_info
                # torch.save(model.state_dict(), './best_{}_appnp.pkl'.format(dataset.name))
            best_val_acc = np.max((best_val_acc, eval_info['val_acc']))
            best_val_loss = np.min((best_val_loss, eval_info['val_loss']))
            bad_counter = 0
        else:
            bad_counter += 1
            if bad_counter == patience:
                break

        print(eval_info)

