import torch
from torch_geometric.nn import GCNConv, Sequential
from typing import Type, List, Dict, Tuple
from gnn_bvp_solver.fem_trainer.models.gcn_processor import GraphNetMP


class WeightedGraphNetMP(GraphNetMP):
    def __init__(
        self,
        n_gcn_layers: int,
        n_input: int = 128,
        n_hidden: int = 128,
        n_output: int = 128,
        gcn_type: Type = GCNConv,
        gcn_kwargs: Dict = None,
        processor_dropout: bool = False,
    ):
        """Simple graph net with n message passing layers and weighted edges"""
        super().__init__(n_gcn_layers, n_input, n_hidden, n_output, gcn_type, gcn_kwargs, processor_dropout)

    def _get_gcn_layer(self, gcn_type: Type, n_in: int, n_out: int, gcn_kwargs: Dict) -> Tuple:
        return gcn_type(n_in, n_out, **gcn_kwargs), "x, edge_index, edge_weights -> x"

    def _init_processor_layers(self, layers: List) -> torch.nn.Module:
        self.conv_layers = Sequential(
            "x, edge_index, edge_weights",
            layers,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
        """Model forward"""
        return self.conv_layers(x, edge_index, edge_weights)
