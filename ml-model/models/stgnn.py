import torch
import torch.nn as nn

class GATConv(nn.Module):
    """Simple GAT-like layer using pure PyTorch — no torch_geometric needed."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.attn   = nn.Linear(2 * out_channels, 1)
        self.act    = nn.ELU()

    def forward(self, x, edge_index):
        x = self.linear(x)
        row, col = edge_index
        alpha = torch.sigmoid(self.attn(torch.cat([x[row], x[col]], dim=-1)))
        out = torch.zeros_like(x)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(x[col]), alpha * x[col])
        return self.act(out + x)

class STGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.fc    = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(self.fc(x))
