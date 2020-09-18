from kernel.model.layers import GraphSpectralFilterLayer
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, global_add_pool, global_sort_pool, global_max_pool


class Decimation(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(Decimation, self).__init__()
        self.conv1 = GraphSpectralFilterLayer(dataset.num_node_features, hidden,
                                                 dropout=0.6, out_channels=8, pre_training=False,
                                                 alpha=0.2, chebyshev_order=14, concat=False)
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers - 1):
            self.convs.append(GraphSpectralFilterLayer(hidden, hidden,
                                  dropout=0.6, out_channels=8, pre_training=False,
                                  alpha=0.2, chebyshev_order=14, concat=False))

        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
