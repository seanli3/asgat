# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from scipy.sparse import coo_matrix
from scipy import sparse
from torch_sparse import spspmm


class Graph(object):
    def __init__(self, data, lap_type="normalized"):
        self.data = data
        self.n_vertices = data.num_nodes
        self.lap_type = lap_type
        self.L = None
        self._lmax = None
        self._dw = None
        self._lmax_method = None

        self.compute_laplacian(lap_type)

    def _get_upper_bound(self):
        return 2.0  # Equal iff the graph is bipartite.

    def compute_laplacian(self, lap_type='combinatorial', q=0.02):
        if lap_type != self.lap_type:
            # Those attributes are invalidated when the Laplacian is changed.
            # Alternative: don't allow the user to change the Laplacian.
            self._lmax = None

        self.lap_type = lap_type

        d = torch.pow(self.dw, -0.5)
        diagonal_indices = torch.tensor([list(range(self.n_vertices)), list(range(self.n_vertices))])
        A = torch.sparse_coo_tensor(self.data.edge_index.to('cpu'), torch.ones(self.data.num_edges, device='cpu')).coalesce()
        indexDmA, valueDmA = spspmm(
            d.indices().repeat(2, 1),
            d.values(),
            # d.values().double(),
            A.indices(),
            A.values(),
            # self.A.values().double(),
            self.n_vertices,
            self.n_vertices,
            self.n_vertices,
        )
        indexDmAmD, valueDmAmD = spspmm(
            indexDmA,
            valueDmA,
            d.indices().repeat(2, 1),
            # d.values().double(),
            d.values(),
            self.n_vertices,
            self.n_vertices,
            self.n_vertices
        )
        self.L = (torch.sparse_coo_tensor(diagonal_indices, torch.ones(self.n_vertices),
                                         [self.n_vertices, self.n_vertices]) \
                 - torch.sparse_coo_tensor(indexDmAmD, valueDmAmD, [self.n_vertices, self.n_vertices])).to_dense()

        self.L.to(self.data.x.device)

    @property
    def dw(self):
        if self._dw is None:
            A = torch.sparse_coo_tensor(self.data.edge_index.to('cpu'), torch.ones(self.data.num_edges, device='cpu')).coalesce()
            self._dw = torch.sparse.sum(A, dim=0).float().to('cpu')
        return self._dw

    @property
    def lmax(self):
        if self._lmax is None:
            self.estimate_lmax()
        return self._lmax

    def estimate_lmax(self, method='lanczos'):
        if method == self._lmax_method:
            return
        self._lmax_method = method

        if method == 'lanczos':
            try:
                # We need to cast the matrix L to a supported type.
                # TODO: not good for memory. Cast earlier?
                L = self.L.to('cpu').to_sparse()
                L_coo = coo_matrix((L.values().numpy(), L.indices().numpy()))
                lmax = sparse.linalg.eigsh(L_coo.asfptype(), k=1, tol=5e-3,
                             ncv=min(self.n_vertices, 10),
                             # return_eigenvectors=False).astype('float16')
                             return_eigenvectors = False)
                lmax = lmax[0]
                if lmax > self._get_upper_bound() + 1e-12:
                    lmax = 2
                lmax *= 1.01  # Increase by 1% to be robust to errors.
                self._lmax = lmax
            except sparse.linalg.ArpackNoConvergence:
                raise ValueError('The Lanczos method did not converge. '
                                 'Try to use bounds.')
        elif method == 'bounds':
            self._lmax = self._get_upper_bound()
        else:
            raise ValueError('Unknown method {}'.format(method))


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
        twf_cur = (G.L - a2*torch.eye(G.n_vertices)) / a1

        nf = c.shape[1]
        r = torch.empty(nf*G.n_vertices, G.n_vertices)

        tmpN = np.arange(G.n_vertices, dtype=int)
        for i in range(nf):
            r[tmpN + G.n_vertices*i, :] = twf_old * 0.5 * c[0][i] + twf_cur * c[1][i]

        factor = (2 / a1) * (G.L - a2*torch.eye(G.n_vertices))

        for k in range(2, M):
            twf_new = factor.mm(twf_cur) - twf_old
            for i in range(nf):
                r[tmpN + G.n_vertices * i, :] += twf_new*c[k,i]

            twf_old = twf_cur
            twf_cur = twf_new

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
