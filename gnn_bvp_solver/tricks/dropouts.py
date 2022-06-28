import torch
from torch import nn
from torch_geometric.utils import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes


class DropNode(nn.Module):
    """
    DropNode: Sampling node using a uniform distribution.
    Based on https://github.com/VITA-Group/Deep_GCN_Benchmarking
    """

    def __init__(self, drop_rate: float):
        """Set dropout rate rate"""
        super(DropNode, self).__init__()
        self.drop_rate = drop_rate

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        num_nodes: int = None,
    ) -> torch.Tensor:
        """Randomly drop nodes at specified rate"""
        if not self.training:
            return edge_index, edge_attr

        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        nodes = torch.arange(num_nodes, dtype=torch.int64)
        mask = torch.full_like(nodes, 1 - self.drop_rate, dtype=torch.float32)
        mask = torch.bernoulli(mask).to(torch.bool)
        subnodes = nodes[mask]

        return subgraph(subnodes, edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
