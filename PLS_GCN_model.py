import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, add_self_loops
import torch.nn.functional as F
import math
import pickle


# %% PLS layer
class PLS(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, A, K):
        Y = X @ X.T
        # with open('./data/residual_layer2.pkl', 'wb') as f:
        for i in range(K):
            component = (torch.diagonal(torch.matmul(torch.transpose(A, 1, 2), Y), dim1=1, dim2=2).sum(
                dim=1).unsqueeze(1).unsqueeze(2) * A).sum(dim=0)  # shape of component is same as the element of A
            denominator = torch.trace(component.T @ component)
            numerator = torch.trace(torch.matmul(component.T, Y))
            Y = Y - (numerator / denominator) * component
            # pickle.dump(Y, f)

            if i == K:
                break
            else:
                beta = torch.diagonal(torch.matmul(torch.transpose(A, 1, 2), component), dim1=1, dim2=2).sum(
                    dim=1) / denominator
                A = A - beta.unsqueeze(1).unsqueeze(2) * component
        return Y


# %% Loss function
class _LossFunction(torch.nn.Module):
    def __init__(self, weight):
        super(_LossFunction, self).__init__()
        self.weight = weight

    def forward(self, out, target, x, residual):
        ce_loss = F.cross_entropy(out, target)
        matrix_diff = x @ x.T - residual
        frobenius_loss = torch.sum(matrix_diff.pow(2)) / (x.size(0) ** 2)
        loss = ce_loss + self.weight * frobenius_loss
        return loss, ce_loss, frobenius_loss


# %% network architecture
class GCN(torch.nn.Module):
    def __init__(self, class_num, program):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(128, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.classifier = Linear(16, class_num)
        self.LeakyRule = torch.nn.LeakyReLU(0.1)
        self.residual = PLS()
        self.program = program

    def forward(self, data, K, device):
        x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes
        edge_index_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        adj_matrix = to_dense_adj(edge_index_loops, None, None, x.size(0))[0]
        A = torch.empty(K, *adj_matrix.shape).to(device)
        for k in range(K):
            s = torch.matrix_power(adj_matrix, k + 1)
            D = torch.diag(torch.pow(torch.sum(s, dim=1), -0.5))
            A[k, :, :] = torch.matmul(torch.matmul(D, s), D)

        if self.program == 'tanh+none':
            h = self.conv1(x, edge_index)
            h = torch.tanh(h)
            h = self.conv2(h, edge_index)
            h = torch.tanh(h)
            h = self.conv3(h, edge_index)
            sigma_hat = h
            h = torch.tanh(h)  # Final GNN embedding space.

        if self.program == 'tanh':
            h = self.conv1(x, edge_index)
            h = torch.tanh(h)
            h = self.conv2(h, edge_index)
            h = torch.tanh(h)
            h = self.conv3(h, edge_index)
            h = torch.tanh(h)
            sigma_hat = h

        if self.program == 'leakyrule':
            h = self.conv1(x, edge_index)
            h = self.LeakyRule(h)
            h = self.conv2(h, edge_index)
            h = self.LeakyRule(h)
            h = self.conv3(h, edge_index)
            h = torch.tanh(h)
            sigma_hat = h

        out = self.classifier(h)

        residual = self.residual(sigma_hat, A, K)

        return sigma_hat, out, residual
