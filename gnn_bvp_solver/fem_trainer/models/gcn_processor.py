from typing import Dict, Type, List, Tuple
import torch
from torch_geometric.nn import GCNConv, Sequential
from torch.nn import ReLU, Dropout


class GraphNetMP(torch.nn.Module):
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
        """Simple graph net with n message passing layers."""
        super().__init__()

        if gcn_kwargs is None:
            gcn_kwargs = {}

        if processor_dropout:
            layers = [Dropout(0.1), self._get_gcn_layer(gcn_type, n_input, n_hidden, gcn_kwargs), ReLU()]
        else:
            layers = [self._get_gcn_layer(gcn_type, n_input, n_hidden, gcn_kwargs), ReLU()]

        for _i in range(n_gcn_layers - 2):
            if processor_dropout:
                layers.append(Dropout(0.1))

            layers.append(self._get_gcn_layer(gcn_type, n_hidden, n_hidden, gcn_kwargs))
            layers.append(ReLU())

        if processor_dropout:
            layers.append(Dropout(0.1))

        layers.append(self._get_gcn_layer(gcn_type, n_hidden, n_output, gcn_kwargs))
        layers.append(ReLU())
        self._init_processor_layers(layers)

    def _get_gcn_layer(self, gcn_type: Type, n_in: int, n_out: int, gcn_kwargs: Dict) -> Tuple:
        return gcn_type(n_in, n_out, **gcn_kwargs), "x, edge_index -> x"

    def _init_processor_layers(self, layers: List) -> torch.nn.Module:
        self.conv_layers = Sequential(
            "x, edge_index",
            layers,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Model forward"""
        return self.conv_layers(x, edge_index)
