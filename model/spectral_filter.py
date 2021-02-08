# -*- coding: utf-8 -*-
import torch
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
        self._e = None
        self._U = None

        self.compute_laplacian(lap_type)

    def _get_upper_bound(self):
        return 2.0  # Equal iff the graph is bipartite.

    def compute_laplacian(self, lap_type='normalized', q=0.02):
        diagonal_indices = torch.tensor([list(range(self.n_vertices)), list(range(self.n_vertices))])
        A = torch.sparse_coo_tensor(self.data.edge_index.to('cpu'),
                                    torch.ones(self.data.num_edges, device='cpu')).coalesce()
        if lap_type == 'normalized':
            if lap_type != self.lap_type:
                # Those attributes are invalidated when the Laplacian is changed.
                # Alternative: don't allow the user to change the Laplacian.
                self._lmax = None

            self.lap_type = lap_type

            d = torch.pow(self.dw, -0.5)
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
        elif lap_type == 'combinatorial':
            D = torch.sparse_coo_tensor(diagonal_indices, self.dw.to_dense(), [self.n_vertices, self.n_vertices], \
                                        device=self.data.x.device)
            self.L = (D - A).to_dense()
        else:
            self.L = A.to_dense()

    @property
    def dw(self):
        if self._dw is None:
            A = torch.sparse_coo_tensor(self.data.edge_index.to('cpu'), torch.ones(self.data.num_edges, device='cpu')).coalesce()
            self._dw = torch.sparse.sum(A, dim=0).float().to('cpu')
        return self._dw

    @property
    def U(self):
        if self.n_vertices > 1000:
            raise NotImplementedError('Eigen-decomposition is not available for large graphs')
        if self._U is None:
            self._e, self._U = torch.symeig(self.L, eigenvectors=True)
        return self._U

    @property
    def e(self):
        if self.n_vertices > 1000:
            raise NotImplementedError('Eigen-decomposition is not available for large graphs')
        if self._e is None:
            self._e, self._U = torch.symeig(self.L, eigenvectors=True)
        return self._e

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
    def __init__(self, G, nf, device, order=32, method="chebyshev", sample_size=300, Kb=18, Ka=2, Tmax=200, kernel=None):
        super(Filter, self).__init__()
        self.G = G
        self.method=method
        self.device = device
        self.nf = nf

        self.order = order
        if method.lower() == 'chebyshev':
            self.cheby_weight = nn.Parameter(torch.empty(self.order+1, self.nf, device=self.device))
        if method.lower() == 'lanzcos':
            self.lanzcos_weight = nn.Parameter(torch.empty(self.G.n_vertices, self.order, self.nf, device=self.device))
            signal = torch.eye(self.G.n_vertices, device=self.device).view(self.G.n_vertices, self.G.n_vertices, 1)
            V, H, _ = self.lanczos(
                self.G.L,
                self.order,
                signal
            )
            Eh, Uh = torch.symeig(H, eigenvectors=True)
            Eh[Eh < 0] = 0
            self.V = torch.matmul(V, Uh)
        elif method.lower() == 'arma':
            self.Ka = Ka
            self.Kb = Kb
            self.Tmax = Tmax
            self.sample_size = sample_size
            self.ia = nn.Parameter(torch.empty(self.nf, Ka, 1, device=self.device))
            self.ib = nn.Parameter(torch.empty(self.nf, Kb + 1, 1, device=self.device))
        elif method.lower() == 'exact':
            self._kernel = kernel

    def reset_parameters(self):
        if self.method.lower() == 'chebyshev':
            nn.init.xavier_normal_(self.cheby_weight)
        elif self.method.lower() == 'lanzcos':
            nn.init.xavier_normal_(self.lanzcos_weight)
        elif self.method.lower() == 'arma':
            nn.init.xavier_normal_(self.ia)
            nn.init.xavier_normal_(self.ib)

    def cheby_op(self) -> torch.Tensor:
        c = self.cheby_weight
        G = self.G
        M = c.shape[0]

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

    def lanczos_op(self):
        signal = torch.eye(self.G.n_vertices, device=self.device).view(self.G.n_vertices, self.G.n_vertices, 1)
        c = self.V.matmul(self.lanzcos_weight * self.V.permute(0, 2, 1).matmul(signal))
        return c.view(self.G.n_vertices, self.G.n_vertices*self.nf).T

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
        if self.method.lower() == "chebyshev":
            s = self.cheby_op()
        elif self.method.lower() == 'lanzcos':
            s = self.lanczos_op()
        elif self.method.lower() == 'arma':
            a = torch.cat([torch.ones(self.nf, 1, 1, device=self.device), self.ia], dim=1)
            b = self.ib
            s = self.agsp_filter_ARMA_cgrad(b, a, Tmax=self.Tmax)
        elif self.method.lower() == 'exact':
            signal = torch.eye(self.G.n_vertices, device=self.device).view(self.G.n_vertices, self.G.n_vertices, 1)
            s_hat = self.G.U.matmul(signal)
            s_hat = s_hat.matmul(self._kernel(self.G.e).view(self.G.n_vertices, 1, self.nf)).permute(2, 0, 1)
            s = self.G.U.matmul(s_hat).view(-1, self.G.n_vertices)

        return s

    def agsp_filter_ARMA_cgrad(self, b, a, tol=1e-4, Tmax=200):
        # For stability, we will work with a shifted version of the Laplacian
        M = 0.5 * self.G.lmax * torch.eye(self.G.n_vertices, device=self.device) - self.G.L
        # M = self.G.L
        x = torch.eye(self.G.n_vertices, device=self.device).view(self.G.n_vertices, self.G.n_vertices, 1)
        b = L_mult(M, b, x, len(x.shape) >= 3)
        y0 = b
        y = y0
        r = b - L_mult(M, a, y, len(x.shape) >= 3)
        p = r
        if len(r.shape) < 3:
            rsold = r.T@r
        elif len(r.shape) == 3:
            rsold = r.permute(0, 2, 1)@r
        else:
            rsold = r.permute(0, 1, 3, 2)@r
        for k in range(Tmax):
            Ap = L_mult(M, a, p, len(x.shape) >= 3)
            if len(p.shape) < 3:
                alpha = rsold / ((p.T @ Ap)+1e-12)
            elif len(r.shape) == 3:
                alpha = rsold / ((p.permute(0, 2, 1) @ Ap)+1e-12)
            else:
                alpha = rsold / ((p.permute(0, 1, 3, 2) @ Ap)+1e-12)
            alpha.clamp_(min=-9e12, max=9e12)
            y = y + alpha * p
            assert not torch.isnan(y).any()
            r = r - alpha * Ap
            if len(r.shape) < 3:
                rsnew = r.T@r
            elif len(r.shape) == 3:
                rsnew = r.permute(0, 2, 1) @ r
            else:
                rsnew = r.permute(0, 1, 3, 2) @ r

            if (rsnew.sqrt() <= tol).any():
                break
            else:
                p = r + (rsnew / rsold) * p
                rsold = rsnew
        return y.view(self.nf * self.G.n_vertices, self.G.n_vertices)


    def agsp_filter_ARMA(self, b, a, tol=1e-4, Tmax=200):
        # For stability, we will work with a shifted version of the Laplacian
        M = 0.5 * self.G.lmax * torch.eye(self.G.n_vertices, device=self.device) - self.G.L
        # M = self.G.L
        x = torch.eye(self.G.n_vertices, device=self.device)

        a = a/a[:, 0, :].view(-1, 1, 1)
        b = b/a[:, 0, :].view(-1, 1, 1)

        y = torch.zeros(self.nf, self.G.n_vertices, self.G.n_vertices, Tmax, device=self.device)
        for t in range(Tmax):
            old_y = torch.zeros(self.nf, self.G.n_vertices, self.G.n_vertices, device=self.device)
            for k in range(self.Ka):
                if t > 0:
                    if k == 0:
                        z = old_y
                    z = M.matmul(z)
                    y = y - a[:, k+1, :].view(-1, 1, 1)*z

            z = x
            for k in range(-1, self.Kb):
                y = y + b[:, k+1, :].view(-1, 1, 1)*z
                z = M.matmul(z)

            old_y = y

            if t > 0 and (torch.norm(y - old_y)/torch.norm(old_y) < tol).any():
                break

        return y.view(self.nf * self.G.n_vertices, self.G.n_vertices)

def kron(m1, m2):
    matrix1 = m1 if len(m1.shape) ==2 else m1.view(-1, 1)
    matrix2 = m2 if len(m2.shape) ==2 else m2.view(-1, 1)
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute([0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))

def L_mult(L, coef, x, expand=False):
    dims = [-1, 1, 1, 1] if expand else [-1, 1, 1]
    y = coef[:, 0].view(*dims) * x
    for i in range(1, coef.shape[1]):
        x = L @ x
        y = y + (coef[:, i].view(*dims) * x)

    return y
