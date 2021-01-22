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
                                         [self.n_vertices, self.n_vertices], device=self.data.x.device) \
                 - torch.sparse_coo_tensor(indexDmAmD, valueDmAmD, [self.n_vertices, self.n_vertices], device=self.data.x.device)).to_dense()

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
                L = self.L.cpu().to_sparse()
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
    def __init__(self, G, kernel, nf, device, order=32, method="chebyshev"):
        super(Filter, self).__init__()
        self.G = G
        self.method=method
        self.device = device
        self.nf = nf

        self._kernel = kernel
        self.order = order

    def evaluate(self, x):
        y = self._kernel(x)
        return y

    def compute_cheby_coeff(self, m: int = 32, N: int = None) -> torch.Tensor:
        if not N:
            N = m + 1

        a_arange = [0, self.G.lmax]

        a1 = (a_arange[1] - a_arange[0]) / 2
        a2 = (a_arange[1] + a_arange[0]) / 2

        tmpN = torch.arange(N, device=self.device)
        num = torch.cos(np.pi * (tmpN + 0.5) / N)
        c = 2. / N * torch.cos(np.pi * tmpN.view(-1,1) * (tmpN + 0.5) / N).mm(self._kernel(a1 * num + a2))

        return c

    def cheby_op(self, c: torch.Tensor) -> torch.Tensor:
        G = self.G
        M = c.shape[0]

        if M < 2:
            raise TypeError("The coefficients have an invalid shape")

        # thanks to that, we can also have 1d signal.

        a_arange = [0, G.lmax]

        a1 = float(a_arange[1] - a_arange[0]) / 2.
        a2 = float(a_arange[1] + a_arange[0]) / 2.

        twf_old = torch.eye(G.n_vertices, device=self.device)
        twf_cur = (G.L - a2*torch.eye(G.n_vertices, device=self.device)) / a1

        nf = c.shape[1]
        r = (twf_old*0.5).multiply(c[0, :nf].repeat(G.n_vertices, G.n_vertices, 1).T) \
               + twf_cur.multiply(c[1, :nf].repeat(G.n_vertices, G.n_vertices, 1).T)
        r = r.reshape(nf*G.n_vertices, G.n_vertices)

        factor = (2 / a1) * (G.L - a2*torch.eye(G.n_vertices, device=self.device))

        for k in range(2, M):
            twf_new = factor.mm(twf_cur) - twf_old
            r += (twf_new*c[k,:].repeat(G.n_vertices, G.n_vertices, 1).T)\
                .reshape(nf*G.n_vertices, G.n_vertices)

            twf_old = twf_cur
            twf_cur = twf_new

        return r

    def lanczos_op(self, order=16):
        signal = torch.eye(self.G.n_vertices, device=self.device).view(self.G.n_vertices, self.G.n_vertices, 1)
        V, H, _ = self.lanczos(
            self.G.L,
            order,
            signal
        )
        Eh, Uh = torch.symeig(H, eigenvectors=True)
        Eh[Eh < 0] = 0
        V = torch.matmul(V, Uh)
        nf = self._kernel.out_channel
        fe = self._kernel(Eh)
        c = V.matmul(fe.view(Eh.shape[0], Eh.shape[1], nf) * V.permute(0, 2, 1).matmul(signal))
        return c.view(self.G.n_vertices, self.G.n_vertices*nf).T

    def lanczos(self, A, order, x):
        Z, N, M = x.shape

        # normalization
        # q = torch.divide(x, kron(torch.ones((1, N)), torch.linalg.norm(x, axis=0)))
        # q = x when x is kronecker
        q = x

        # initialization
        hiv = torch.arange(0, order * M, order, device=self.device)
        V = torch.zeros((Z, N, M * order), device=self.device)
        V[:, :, hiv] = q

        H = torch.zeros((Z, order + 1, M * order), device=self.device)
        r = torch.matmul(A, q)
        H[:, 0, hiv] = torch.sum(q * r, axis=1)
        # r -= (kron(torch.ones((N, 1)), H[0, hiv].view(1, -1))) * q
        # (kron(torch.ones((N, 1)), H[0, hiv].view(1, -1))) will always be all ones
        r -= q
        H[:, 1, hiv] = torch.linalg.norm(r, axis=1)

        orth = torch.zeros(Z, order, device=self.device)
        orth[:, 0] = torch.linalg.norm(torch.matmul(V.permute(0, 2, 1), V) - M, axis=(1, 2))

        for k in range(1, order):
            if H.isnan().any() or H.isinf().any():
                H = H[:, :k, :k]
                V = V[:, :, :k]
                orth = orth[:, :k]
                return V, H, orth

            H[:, k - 1, hiv + k] = H[:, k, hiv + k - 1]
            v = q
            q = r / (H[:, k - 1, k + hiv]).repeat(1, 1, N).view(N, N, 1)

            V[:, :, k + hiv] = q

            r = torch.matmul(A, q)
            r -= H[:, k - 1, k + hiv].repeat(1, 1, N).view(N, N, 1) * v
            H[:, k, k + hiv] = torch.sum(torch.multiply(q, r), axis=1)
            r -= H[:, k, k + hiv].repeat(1, 1, N).view(N, N, 1) * q

            # The next line has to be checked
            r -= torch.matmul(V, torch.matmul(V.permute(0, 2, 1), r))  # full reorthogonalization
            H[:, k + 1, k + hiv] = torch.linalg.norm(r, axis=1)
            temp = torch.matmul(V.permute(0, 2, 1), V) - M
            orth[:, k] = torch.linalg.norm(temp, axis=(1, 2))

        H = H[:, :order, :order]

        return V, H, orth

    # deprecated
    def lanczos_seq(self, A, order, x):
        N, M = x.shape

        # normalization
        # q = torch.divide(x, kron(torch.ones((1, N)), torch.linalg.norm(x, axis=0)))
        # q = x when x is kronecker
        q = x

        # initialization
        hiv = torch.arange(0, order * M, order, device=self.device)
        V = torch.zeros((N, M * order), device=self.device)
        V[:, hiv] = q

        H = torch.zeros((order + 1, M * order), device=self.device)
        r = torch.matmul(A, q)
        H[0, hiv] = torch.sum(q * r, axis=0)
        # r -= (kron(torch.ones((N, 1)), H[0, hiv].view(1, -1))) * q
        # (kron(torch.ones((N, 1)), H[0, hiv].view(1, -1))) will always be all ones
        r -= q
        H[1, hiv] = torch.linalg.norm(r, axis=0)

        orth = torch.zeros(order, device=self.device)
        orth[0] = torch.linalg.norm(torch.matmul(V.T, V) - M)

        for k in range(1, order):
            if H.isnan().any() or H.isinf().any():
                H = H[:k, :k]
                V = V[:, :k]
                orth = orth[:k]
                return V, H, orth

            H[k - 1, hiv + k] = H[k, hiv + k - 1]
            v = q
            q = r / (H[k - 1, k + hiv]).repeat(N, 1)
            V[:, k + hiv] = q

            r = torch.matmul(A, q)
            r -= H[k - 1, k + hiv].repeat(N, 1) * v
            H[k, k + hiv] = torch.sum(torch.multiply(q, r), axis=0)
            r -= H[k, k + hiv].repeat(N, 1) * q

            # The next line has to be checked
            r -= torch.matmul(V, torch.matmul(V.T, r))  # full reorthogonalization
            H[k + 1, k + hiv] = torch.linalg.norm(r, axis=0)
            temp = torch.matmul(V.T, V) - M
            orth[k] = torch.linalg.norm(temp)

        H = H[:order, :order]

        return V, H, orth

    # deprecated
    def lanczos_op_seq(self, order=16):
        signal = torch.eye(self.G.n_vertices, device=self.device).view(self.G.n_vertices, self.G.n_vertices, 1)
        tmpN = torch.arange(self.G.n_vertices)
        nf = self._kernel.out_channel
        c = torch.zeros((self.G.n_vertices*nf, self.G.n_vertices))
        for j in range(signal.shape[0]):
            V, H, _ = self.lanczos_seq(
                self.G.L,
                order,
                signal[j]
            )
            Eh, Uh = torch.symeig(H, eigenvectors=True)
            Eh[Eh < 0] = 0
            V = torch.matmul(V, Uh)
            fe = self._kernel(Eh)
            for i in range(nf):
                c[tmpN + i*self.G.n_vertices, j] = V.matmul(fe[:,i].view(-1, 1) * V.T.matmul(signal[j])).view(-1)
        return c

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
        if self.method == "chebyshev":
            c = self.compute_cheby_coeff(m=self.order)
            s = self.cheby_op(c)
        else:
            s = self.lanczos_op(order=self.order)

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
        x = torch.linspace(torch.min(self.G.e), self.G.lmax, self.order).detach()
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


def _sum_ind(ind1, ind2):
    ind = ind1.view(-1).repeat(ind2.size(), 1).T + ind2.view(-1)
    return ind.view(-1)


def kron(m1, m2):
    matrix1 = m1 if len(m1.shape) ==2 else m1.view(-1, 1)
    matrix2 = m2 if len(m2.shape) ==2 else m2.view(-1, 1)
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute([0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))
