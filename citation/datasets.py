import os.path as osp

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, dropout_adj
from random import sample
import torch


def get_planetoid_dataset(name, normalize_features=False, transform=None, edge_dropout=None, node_feature_dropout=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    if edge_dropout:
        # edge_list, _ = add_self_loops(dataset.data.edge_index)
        edge_list = dataset.data.edge_index
        num_edges = edge_list.shape[1]
        edge_list, _ = dropout_adj(edge_list, p=edge_dropout, force_undirected=True)
        print('Edge dropout rate: {:.4f}'.format(1 - edge_list.shape[1] / num_edges))
        dataset.data.edge_index = edge_list
    if node_feature_dropout:
        num_nodes = dataset.data.num_nodes
        drop_indices = sample(range(num_nodes), int(node_feature_dropout*num_nodes))
        print('Node feature dropout rate: {:.4f}'.format(len(drop_indices)/num_nodes))
        dataset.data.x.index_fill_(0, torch.tensor(drop_indices), 0)

    return dataset
