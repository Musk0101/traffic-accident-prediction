import torch
from torch_geometric.data import Data

def create_temporal_graphs(X, edge_index, window=5):
    """
    X: [N, F]
    edge_index: [2, E]
    """

    X = torch.tensor(X, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    graphs = []

    for _ in range(195):
        # IMPORTANT: x must stay [N, F]
        y = torch.tensor([X[:, -1].mean() > 2], dtype=torch.float)

        graphs.append(
            Data(
                x=X,
                edge_index=edge_index,
                y=y
            )
        )

    return graphs
