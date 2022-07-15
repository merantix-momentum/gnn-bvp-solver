from torch.nn import Module
import torch


class GNNIdentity(Module):
    def __init__(self, *args, **kwargs):
        """Simple identity function that can be used instead of a GNN processor"""
        super(GNNIdentity, self).__init__()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Model forward"""
        return x
