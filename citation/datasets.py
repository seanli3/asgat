import os.path as osp
from torch_geometric.datasets import Planetoid, PPI, Amazon, Reddit, Coauthor, SNAPDataset, PPI
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, dropout_adj
from random import sample
from torch.nn import functional as F
from torch_geometric.utils import to_networkx
import networkx as nx
import torch
import networkx as nx
from scipy.sparse import coo_matrix
import numpy as np


def matching_labels_distribution(dataset):
    # Build graph
    adj = coo_matrix(
        (np.ones(dataset[0].num_edges),
        (dataset[0].edge_index[0].numpy(), dataset[0].edge_index[1].numpy())),
        shape=(dataset[0].num_nodes, dataset[0].num_nodes))
    G = nx.Graph(adj)

    hop_1_matching_percent = []
    hop_2_matching_percent = []
    hop_3_matching_percent = []
    for n in range(dataset.data.num_nodes):
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

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, dissimilar_mask=None, lcc_mask=None):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        if dissimilar_mask is None and lcc_mask is None:
            index = (data.y == i).nonzero().view(-1)
        elif dissimilar_mask is not None and lcc_mask is None:
            index = (data.y == i).logical_and(dissimilar_mask).nonzero().view(-1)
        else:
            index = (data.y == i).logical_and(dissimilar_mask).logical_and(lcc_mask).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    if len(rest_index) < 60:
        raise Exception('Not enough val and test data')

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:min(len(rest_index)//3, 500)], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[min(len(rest_index)//3, 500):max(len(rest_index)*2//3, 1500)], size=data.num_nodes)

    return data


def random_coauthor_amazon_splits(data, num_classes, lcc_mask):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data


def get_dataset(name, normalize_features=False, transform=None, edge_dropout=None, node_feature_dropout=None,
                dissimilar_t = 1, cuda=False, permute_masks=None, lcc=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    if name in ['Computers', 'Photo']:
        dataset = Amazon(path, name)
    elif name in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(path, name, split="full")
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(path, name, split="full")
    elif name in ['Reddit']:
        dataset = Reddit(path)
    elif name in ['ego-facebook', 'ego-gplus', 'ego-twitter', 'soc-epinions1', 'soc-livejournal1',
                    'soc-pokec', 'soc-slashdot0811', 'soc-slashdot0922', 'wiki-vote']:
        dataset = SNAPDataset(path, name)
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

    if dissimilar_t < 1 and not permute_masks:
        label_distributions = torch.tensor(matching_labels_distribution(dataset)).cpu()
        dissimilar_neighbhour_train_mask = dataset[0]['train_mask']\
            .logical_and(label_distributions[0] <= dissimilar_t)
        dissimilar_neighbhour_val_mask = dataset[0]['val_mask']\
            .logical_and(label_distributions[0] <= dissimilar_t)
        dissimilar_neighbhour_test_mask = dataset[0]['test_mask']\
            .logical_and(label_distributions[0] <= dissimilar_t)
        dataset.data.train_mask = dissimilar_neighbhour_train_mask
        dataset.data.val_mask = dissimilar_neighbhour_val_mask
        dataset.data.test_mask = dissimilar_neighbhour_test_mask

    lcc_mask = None
    if lcc:  # select largest connected component
        data_ori = dataset[0]
        data_nx = to_networkx(data_ori)
        data_nx = data_nx.to_undirected()
        print("Original #nodes:", data_nx.number_of_nodes())
        data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
        print("#Nodes after lcc:", data_nx.number_of_nodes())
        lcc_mask = list(data_nx.nodes)

    if permute_masks is not None:
        label_distributions = torch.tensor(matching_labels_distribution(dataset)).cpu()
        dataset.data = permute_masks(dataset.data, dataset.num_classes, lcc_mask=lcc_mask,
                                     dissimilar_mask=(label_distributions[0] <= dissimilar_t))

    if cuda:
        dataset.data.to('cuda')

    return dataset
