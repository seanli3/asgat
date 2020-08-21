import os.path as osp
from torch_geometric.datasets import Planetoid, PPI, Amazon, Reddit, Coauthor
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, dropout_adj
from random import sample
from torch.nn import functional as F
import torch


def matching_labels_distribution(dataset, nodes_set):
    import networkx as nx
    from scipy.sparse import coo_matrix
    import numpy as np

    # Build graph
    adj = coo_matrix(
        (np.ones(dataset[0].num_edges),
        (dataset[0].edge_index[0].numpy(), dataset[0].edge_index[1].numpy())),
        shape=(dataset[0].num_nodes, dataset[0].num_nodes))
    G = nx.Graph(adj)

    hop_1_matching_percent = []
    hop_2_matching_percent = []
    hop_3_matching_percent = []
    for n in nodes_set:
        hop_1_neighbours = list(nx.ego_graph(G, n, 1).nodes())
        hop_2_neighbours = list(nx.ego_graph(G, n, 2).nodes())
        hop_3_neighbours = list(nx.ego_graph(G, n, 3).nodes())
        node_label = dataset[0].y[n]
        hop_1_labels = dataset[0].y[hop_1_neighbours]
        hop_2_labels = dataset[0].y[hop_2_neighbours]
        hop_3_labels = dataset[0].y[hop_3_neighbours]
        matching_1_labels = node_label == hop_1_labels
        matching_2_labels = node_label == hop_2_labels
        matching_3_labels = node_label == hop_3_labels
        hop_1_matching_percent.append(matching_1_labels.float().sum()/matching_1_labels.shape[0])
        hop_2_matching_percent.append(matching_2_labels.float().sum()/matching_2_labels.shape[0])
        hop_3_matching_percent.append(matching_3_labels.float().sum()/matching_3_labels.shape[0])

    return hop_1_matching_percent, hop_2_matching_percent, hop_3_matching_percent


def get_dataset(name, normalize_features=False, transform=None, edge_dropout=None, node_feature_dropout=None, dissimilar_t = 1, cuda=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    if name in ['Computers', 'Photo']:
        dataset = Amazon(path, name)
    elif name in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(path, name, split="full")
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(path, name, split="full")
    elif name in ['Reddit']:
        dataset = Reddit(path)
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
        drop_indices = sample(list(range(num_nodes)), int(node_feature_dropout * num_nodes))
        dataset.data.x.index_fill_(0, torch.tensor(drop_indices).cpu(), 0)
        print('Node feature dropout rate: {:.4f}' .format(len(drop_indices)/num_nodes))

    dissimilar_neighbhour_train_mask = dataset[0]['train_mask'].clone()
    dissimilar_neighbhour_val_mask = dataset[0]['val_mask'].clone()
    dissimilar_neighbhour_test_mask = dataset[0]['test_mask'].clone()
    label_distributions = matching_labels_distribution(dataset, dissimilar_neighbhour_train_mask.nonzero().view(-1).tolist())
    dissimilar_neighbhour_train_mask[dissimilar_neighbhour_train_mask] = (torch.tensor(label_distributions[0]).cpu() <= dissimilar_t)
    label_distributions = matching_labels_distribution(dataset, dissimilar_neighbhour_val_mask.nonzero().view(-1).tolist())
    dissimilar_neighbhour_val_mask[dissimilar_neighbhour_val_mask] = (torch.tensor(label_distributions[0]).cpu() <= dissimilar_t)
    label_distributions = matching_labels_distribution(dataset, dissimilar_neighbhour_test_mask.nonzero().view(-1).tolist())
    dissimilar_neighbhour_test_mask[dissimilar_neighbhour_test_mask] = (torch.tensor(label_distributions[0]).cpu() <= dissimilar_t)
    dataset.data.train_mask = dissimilar_neighbhour_train_mask
    dataset.data.val_mask = dissimilar_neighbhour_val_mask
    dataset.data.test_mask = dissimilar_neighbhour_test_mask

    if cuda:
        dataset.data.to('cuda')

    return dataset
