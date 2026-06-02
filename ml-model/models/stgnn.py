import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear
from torch_geometric.nn import GATConv

class STGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, T=5):
        super().__init__()

        self.T = T

        self.gat1 = GATConv(in_channels, hidden_channels)
        self.gat2 = GATConv(hidden_channels, hidden_channels)

        self.gru = GRU(hidden_channels, hidden_channels, batch_first=True)
        self.fc = Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        """
        x: [N, F]  (PyTorch Geometric standard)
        """

        N, F_in = x.shape

        # ---- Spatial encoding ----
        h = F.relu(self.gat1(x, edge_index))
        h = F.relu(self.gat2(h, edge_index))  # [N, hidden]

        # ---- Fake temporal sequence ----
        h_seq = h.unsqueeze(1).repeat(1, self.T, 1)  # [N, T, hidden]

        # ---- Temporal modeling ----
        _, h_final = self.gru(h_seq)  # [1, N, hidden]

        out = torch.sigmoid(self.fc(h_final.squeeze(0)))  # [N, 1]
        return out.mean()  # graph-level prediction
