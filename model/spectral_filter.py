# -*- coding: utf-8 -*-
import torch
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.sparse import coo_matrix
import numpy as np
import torch.nn as nn
from torch_sparse import spspmm
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian


class Graph(object):
    def __init__(self, data, lap_type="normalized"):
        self.lap_type = lap_type
        self.data = data
        edge_index, edge_weight = remove_self_loops(data.edge_index)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight, 'sym', data.x.dtype, data.num_nodes)
        lambda_max = torch.tensor(2.0, dtype=data.x.dtype, device=data.x.device)

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=data.num_nodes)
        assert edge_weight is not None

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_vertices = data.num_nodes
        self.lmax = lambda_max.item()


class Filter(nn.Module):
    def __init__(self, G, kernel, chebyshev_order=32):
        super(Filter, self).__init__()
        self.G = G

        self._kernel = kernel
        self.chebyshev_order = chebyshev_order

    def evaluate(self, x):
        y = self._kernel(x)
        return y

    def compute_cheby_coeff(self, m: int = 32, N: int = None) -> torch.Tensor:
        if not N:
            N = m + 1

        a_arange = [0, self.G.lmax]

        a1 = (a_arange[1] - a_arange[0]) / 2
        a2 = (a_arange[1] + a_arange[0]) / 2
        c = []

        tmpN = torch.arange(N)
        num = torch.cos(np.pi * (tmpN + 0.5) / N)
        for o in range(m + 1):
            c.append(2. / N * torch.cos(np.pi * o * (tmpN + 0.5) / N).view(1, -1).mm(self._kernel(a1 * num + a2)))

        return torch.cat(c)

    def cheby_op(self, c: torch.Tensor) -> torch.Tensor:
        G = self.G
        M = c.shape[0]

        if M < 2:
            raise TypeError("The coefficients have an invalid shape")

        # thanks to that, we can also have 1d signal.

        a_arange = [0, G.lmax]

        a1 = float(a_arange[1] - a_arange[0]) / 2.
        a2 = float(a_arange[1] + a_arange[0]) / 2.

        twf_old = torch.eye(G.n_vertices)
        twf_cur = torch.zeros(G.n_vertices, G.n_vertices).index_put((G.edge_index[0], G.edge_index[1]), G.edge_weight)
        twf_cur = (twf_cur - a2*torch.eye(G.n_vertices)) / a1

        nf = c.shape[1]
        r = []

        for i in range(nf):
            r.append(twf_old.to_sparse() * 0.5 * c[0][i] + twf_cur.to_sparse() * c[1][i])

        factor = (2 / a1) * (
                torch.sparse_coo_tensor(G.edge_index, G.edge_weight, (G.n_vertices, G.n_vertices)) -
                torch.sparse_coo_tensor(
                    [range(G.n_vertices), range(G.n_vertices)], torch.ones(G.n_vertices)*a2, [G.n_vertices, G.n_vertices]
                )
        )

        for k in range(2, M):
            twf_new = factor.mm(twf_cur) - twf_old
            for i in range(nf):
                r[i] = twf_new.to_sparse()*c[k,i] + r[i]

            twf_old = twf_cur
            twf_cur = twf_new

        for i in range(len(r)):
            r[i] = r[i].coalesce()

        return r

    def forward(self) -> object:
        """
        Parameters
        ----------
        s: N x Df x Ns. where N is the number of nodes, Nf is the dimension of node features, Ns is the number of signals

        Returns
        -------
        s: torch.Tensor
        """

        # TODO: update Chebyshev implementation (after 2D filter banks).
        c = self.compute_cheby_coeff(m=self.chebyshev_order)
        s = self.cheby_op(c)

        return s

    def localize(self, i, **kwargs):
        s = torch.zeros(self.G.n_vertices)
        s[i] = 1

        return np.sqrt(self.G.n_vertices) * self.forward(s, **kwargs)

    def estimate_frame_bounds(self, x=None):
        if x is None:
            x = torch.linspace(0, self.G.lmax, 1000)

        sum_filters = torch.sum(self.evaluate(x) ** 2, dim=0)

        return sum_filters.min(), sum_filters.max()

    def complement(self, frame_bound=None):
        def kernel(x):
            y = self.evaluate(x)
            y = torch.pow(y, 2)
            y = torch.sum(y, dim=1)

            if frame_bound is None:
                bound = y.max()
            elif y.max() > frame_bound:
                raise ValueError('The chosen bound is not feasible. '
                                 'Choose at least {}.'.format(y.max()))
            else:
                bound = frame_bound

            return torch.sqrt(bound - y)

        return Filter(self.G, kernel)

    def inverse(self):
        A, _ = self.estimate_frame_bounds()
        if A == 0:
            raise ValueError('The filter bank is not invertible as it is not '
                             'a frame (lower frame bound A=0).')

        def kernel(g, i, x):
            y = g.evaluate(x).T
            z = torch.pinverse(y.view(-1, 1)).view(-1)
            return z[:, i]  # Return one filter.

        return Filter(self.G, kernel)

    def plot(self, eigenvalues=None, sum=None, title=None,
             ax=None, **kwargs):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x = torch.linspace(torch.min(self.G.e), self.G.lmax, self.chebyshev_order).detach()
        y = self.evaluate(x).T.detach()
        x = x.cpu()
        y = y.cpu()
        lines = ax.plot(x, y)

        if len(y.shape) == 2:
            for i in range(y.shape[1]):
                ax.plot(x, y[:, i], '.')
        else:
            ax.plot(x, y, '.')
        if sum:
            ax.plot(x, np.sum(y ** 2, 1), '.')

        # TODO: plot highlighted eigenvalues
        if sum:
            line_sum, = ax.plot(x, np.sum(y ** 2, 1), 'k', **kwargs)

        ax.set_xlabel(r"$\lambda$: laplacian's eigenvalues / graph frequencies")
        ax.set_ylabel(r'$\hat{g}(\lambda)$: filter response')
        plt.show()
