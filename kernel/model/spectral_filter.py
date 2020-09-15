# -*- coding: utf-8 -*-
import torch
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.sparse import coo_matrix
import numpy as np
import torch.nn as nn
from torch_sparse import spspmm


class Graph(object):
    def __init__(self, A: torch.Tensor, lap_type="normalized"):
        self.A = A.float().coalesce()
        self.lap_type = lap_type
        self.n_vertices = self.A.shape[0]
        self.n_edges = len(self.A.values())
        self.L = None
        self._lmax = None
        self._D = None
        self._dw = None
        self._lmax_method = None

        self.compute_laplacian(lap_type)

    def _get_upper_bound(self):
        if self.lap_type == 'normalized':
            return 2.0  # Equal iff the graph is bipartite.
        elif self.lap_type == 'combinatorial':
            bounds = []
            # Equal for full graphs.
            bounds += [self.n_vertices * torch.max(self.A)]
            # Gershgorin circle theorem. Equal for regular bipartite graphs.
            # Special case of the below bound.
            bounds += [2 * torch.max(self.dw)]
            # Anderson, Morley, Eigenvalues of the Laplacian of a graph.
            # Equal for regular bipartite graphs.
            if self.n_edges > 0:
                sources, targets = self.A.nonzero(as_tuple=True)
                bounds += [torch.max(self.dw[sources] + self.dw[targets])]
            m = self.A.dot(self.dw) / self.dw  # Mean degree of adjacent vertices.
            bounds += [torch.max(self.dw + m)]
            # Good review: On upper bounds for Laplacian graph eigenvalues.
            return min(bounds)
        elif self.lap_type == 'hermitian':
            return torch.max(torch.symeig(self.L)[0])
        elif self.lap_type == 'random walk':
            return torch.max(1 - torch.eig(self.L)[0].T[0])
        else:
            raise ValueError('Unknown Laplacian type '
                             '{}'.format(self.lap_type))

    def compute_laplacian(self, lap_type='combinatorial', q=0.02):
        if lap_type != self.lap_type:
            # Those attributes are invalidated when the Laplacian is changed.
            # Alternative: don't allow the user to change the Laplacian.
            self._lmax = None
            self._D = None

        self.lap_type = lap_type

        if lap_type == 'combinatorial':
            self.L = self.D - self.A
        elif lap_type == 'normalized':
            d = torch.pow(self.dw, -0.5)
            diagonal_indices = torch.tensor([list(range(self.n_vertices)), list(range(self.n_vertices))])
            indexDmA, valueDmA = spspmm(
                d.indices().repeat(2, 1),
                d.values(),
                self.A.indices(),
                self.A.values(),
                self.n_vertices,
                self.n_vertices,
                self.n_vertices,
            )
            indexDmAmD, valueDmAmD = spspmm(
                indexDmA,
                valueDmA,
                d.indices().repeat(2, 1),
                d.values(),
                self.n_vertices,
                self.n_vertices,
                self.n_vertices
            )
            self.L = torch.sparse_coo_tensor(diagonal_indices, torch.ones(self.n_vertices),
                                             [self.n_vertices, self.n_vertices]) \
                     - torch.sparse_coo_tensor(indexDmAmD, valueDmAmD, [self.n_vertices, self.n_vertices])
            self.L = self.L.coalesce()
        elif lap_type == 'hermitian':
            sym_weighted_adajacency_matrix = (self.A + self.A.T) / 2
            Gamma_q = torch.pow(np.e, 2j * np.pi * q * (self.A - self.A.T))
            self.L = self.D - Gamma_q.mm(sym_weighted_adajacency_matrix)
        elif lap_type == 'random walk':
            self.L = torch.inverse(torch.diag(self.dw)).mm(self.A)
        else:
            raise ValueError('Unknown Laplacian type {}'.format(lap_type))

    @property
    def dw(self):
        if self._dw is None:
            self._dw = torch.sparse.sum(self.A, dim=0).float()
        return self._dw

    @property
    def lmax(self):
        if self._lmax is None:
            self.estimate_lmax()
        return self._lmax

    @property
    def D(self):
        if self._D is None:
            self._D = torch.sparse_coo_tensor([range(self.n_vertices), range(self.n_vertices)],
                                        self.dw.values(), [self.n_vertices, self.n_vertices])
        return self._D

    def estimate_lmax(self, method='lanczos'):
        if method == self._lmax_method:
            return
        self._lmax_method = method

        if method == 'lanczos':
            try:
                # We need to cast the matrix L to a supported type.
                # TODO: not good for memory. Cast earlier?
                L_coo = coo_matrix((self.L.values().cpu(), self.L.indices().cpu().numpy()))
                lmax = eigsh(L_coo.asfptype(), k=1, tol=5e-3,
                             ncv=min(self.n_vertices, 10),
                             return_eigenvectors=False)
                lmax = lmax[0]
                if lmax > self._get_upper_bound() + 1e-12:
                    lmax = 2
                lmax *= 1.01  # Increase by 1% to be robust to errors.
                self._lmax = lmax
            except ArpackNoConvergence:
                raise ValueError('The Lanczos method did not converge. '
                                 'Try to use bounds.')
        elif method == 'bounds':
            self._lmax = self._get_upper_bound()
        else:
            raise ValueError('Unknown method {}'.format(method))


class Filter(nn.Module):
    def __init__(self, kernel, chebyshev_order=32):
        super(Filter, self).__init__()

        self._kernel = kernel
        self.chebyshev_order = chebyshev_order


    def evaluate(self, x):
        y = self._kernel(x)
        return y

    def cheby_eval(self, x, G):
        x = x.view(-1, 1)
        a, b = 0, G.lmax
        y = (2.0 * x - a - b) * (1.0 / (b - a))
        y2 = 2.0 * y
        c = self.compute_cheby_coeff(G, m=self.chebyshev_order)
        (d, dd) = (c[-1], 0)  # Special case first step for efficiency
        for cj in c.flip(0)[1:-1]:  # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * c[0]

    def compute_cheby_coeff(self, G, m: int = 32, N: int = None) -> torch.Tensor:
        if not N:
            N = m + 1

        a_arange = [0, G.lmax]

        a1 = (a_arange[1] - a_arange[0]) / 2
        a2 = (a_arange[1] + a_arange[0]) / 2
        c = []

        tmpN = torch.arange(N)
        num = torch.cos(np.pi * (tmpN + 0.5) / N)
        for o in range(m + 1):
            c.append(2. / N * torch.cos(np.pi * o * (tmpN + 0.5) / N).view(1, -1).mm(self._kernel(a1 * num + a2)))

        return torch.cat(c)

    def cheby_op(self, c: torch.Tensor, G) -> torch.Tensor:
        M = c.shape[0]

        if M < 2:
            raise TypeError("The coefficients have an invalid shape")

        # thanks to that, we can also have 1d signal.


        signal = torch.sparse_coo_tensor(
            [range(G.n_vertices), range(G.n_vertices)],
            torch.ones(G.n_vertices),
            [G.n_vertices, G.n_vertices]
        ).coalesce()

        a_arange = [0, G.lmax]

        a1 = float(a_arange[1] - a_arange[0]) / 2.
        a2 = float(a_arange[1] + a_arange[0]) / 2.

        twf_old = signal
        twf_cur = ((G.L - a2 * signal) / a1).coalesce()

        nf = c.shape[1]
        r = []

        for i in range(nf):
            r.append(
                torch.sparse_coo_tensor(twf_old.indices(), twf_old.values() * 0.5 * c[0][i], twf_old.shape)
                + torch.sparse_coo_tensor(twf_cur.indices(), twf_cur.values() * c[1][i], twf_cur.shape)
            )

        factor = (2 / a1 * (G.L - a2 * signal)).coalesce()

        fmt_index, fmt_value = spspmm(
            factor.indices(),
            factor.values(),
            twf_cur.indices(),
            twf_cur.values(),
            G.n_vertices,
            G.n_vertices,
            G.n_vertices)

        for k in range(2, M):
            twf_new = (torch.sparse_coo_tensor(fmt_index, fmt_value, [G.n_vertices, G.n_vertices]) - twf_old).coalesce()
            for i in range(nf):
                r[i] = torch.sparse_coo_tensor(twf_new.indices(), twf_new.values() * c[k,i], twf_new.shape) + r[i]

            twf_old = twf_cur
            twf_cur = twf_new

        tmp = np.sqrt(G.n_vertices)
        for i in range(len(r)):
            r[i] = r[i].coalesce() * tmp
        return r

    def forward(self, G) -> object:
        """
        Parameters
        ----------
        s: N x Df x Ns. where N is the number of nodes, Nf is the dimension of node features, Ns is the number of signals

        Returns
        -------
        s: torch.Tensor
        """

        # TODO: update Chebyshev implementation (after 2D filter banks).
        c = self.compute_cheby_coeff(G, m=self.chebyshev_order)
        s = self.cheby_op(c, G)

        return s

    def estimate_frame_bounds(self, G, x=None):
        if x is None:
            x = torch.linspace(0, G.lmax, 1000)

        sum_filters = torch.sum(self.evaluate(x) ** 2, dim=0)

        return sum_filters.min(), sum_filters.max()

    def complement(self, G, frame_bound=None):
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

        return Filter(G, kernel)

    def inverse(self, G):
        A, _ = self.estimate_frame_bounds()
        if A == 0:
            raise ValueError('The filter bank is not invertible as it is not '
                             'a frame (lower frame bound A=0).')

        def kernel(g, i, x):
            y = g.evaluate(x).T
            z = torch.pinverse(y.view(-1, 1)).view(-1)
            return z[:, i]  # Return one filter.

        return Filter(G, kernel)

    def plot(self, G, eigenvalues=None, sum=None, title=None,
             ax=None, **kwargs):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x = torch.linspace(torch.min(G.e), G.lmax, self.chebyshev_order).detach()
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
