from typing import List
from gnn_bvp_solver.fem_trainer.models.gcn_processor import GraphNetMP
from gnn_bvp_solver.fem_trainer.models.gcn_processor_weights import WeightedGraphNetMP
from gnn_bvp_solver.fem_trainer.models.gnn_identity import GNNIdentity
from gnn_bvp_solver.fem_trainer.models.two_layer_mlp import TwoLayerMLP
from torch_geometric.nn import GraphUNet, ChebConv, GCN2Conv
from torch_geometric.utils import dropout_adj
from torch_geometric.utils import coalesce, to_undirected
from gnn_bvp_solver.tricks.dropouts import DropNode

import torch.nn.functional as F
import torch


class MainModel(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        mlp_hidden: int,
        processor_hidden: int,
        dim_out: int,
        processor: str,
        augmentation: List,
        remove_pos: bool,
    ):
        """Main model consisting of encoder - processor - decoder"""
        super().__init__()

        self.remove_pos = remove_pos
        if self.remove_pos:
            dim_in -= 2

        self.pass_edge_weights = False
        self.augmentation = [] if augmentation is None else augmentation
        processor_dropout = "processor_dropout" in self.augmentation

        if processor == "unet3":
            self.processor = GraphUNet(mlp_hidden, processor_hidden, mlp_hidden, depth=3)
        elif processor == "unet6":
            self.processor = GraphUNet(mlp_hidden, processor_hidden, mlp_hidden, depth=6)
        elif processor == "gcn9":
            self.processor = GraphNetMP(
                9, mlp_hidden, processor_hidden, mlp_hidden, processor_dropout=processor_dropout
            )
        elif processor == "gcn18":
            self.processor = GraphNetMP(
                18, mlp_hidden, processor_hidden, mlp_hidden, processor_dropout=processor_dropout
            )
        elif processor == "gcn9w":
            self.processor = WeightedGraphNetMP(
                9, mlp_hidden, processor_hidden, mlp_hidden, processor_dropout=processor_dropout
            )
            self.pass_edge_weights = True
        elif processor == "gcnii9w":
            self.processor = WeightedGraphNetMP(
                9, mlp_hidden, processor_hidden, mlp_hidden, gcn_type=GCN2Conv, processor_dropout=processor_dropout
            )
            self.pass_edge_weights = True
        elif processor == "gcnch9w":
            self.processor = WeightedGraphNetMP(
                9,
                mlp_hidden,
                processor_hidden,
                mlp_hidden,
                gcn_type=ChebConv,
                gcn_kwargs={"K": 5},
                processor_dropout=processor_dropout,
            )
            self.pass_edge_weights = True
        elif processor == "gcnch3w":
            self.processor = WeightedGraphNetMP(
                3,
                mlp_hidden,
                processor_hidden,
                mlp_hidden,
                gcn_type=ChebConv,
                gcn_kwargs={"K": 5},
                processor_dropout=processor_dropout,
            )
            self.pass_edge_weights = True
        elif processor == "gcnch3w_noew":
            self.processor = GraphNetMP(
                3,
                mlp_hidden,
                processor_hidden,
                mlp_hidden,
                gcn_type=ChebConv,
                gcn_kwargs={"K": 5},
                processor_dropout=processor_dropout,
            )
        elif processor == "none":
            self.processor = GNNIdentity()
        else:
            raise ValueError("processor")

        self.encoder = TwoLayerMLP(dim_in, mlp_hidden, mlp_hidden)
        self.decoder = TwoLayerMLP(mlp_hidden, mlp_hidden, dim_out, is_output=True)

        if "drop_nodes" in self.augmentation:
            self.drop_nodes = DropNode(0.1)

    def compute_edge_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute edge weights as relative distances between nodes."""
        nodes_s = x[edge_index[0]]
        nodes_t = x[edge_index[1]]

        x_dif = nodes_s[:, 0] - nodes_t[:, 0]
        y_dif = nodes_s[:, 1] - nodes_t[:, 1]

        return torch.sqrt(torch.square(x_dif) + torch.square(y_dif))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # should be fast here as is on gpu
        if self.training:
            edge_index = coalesce(to_undirected(edge_index))

        if "drop_edges" in self.augmentation and self.training:
            # dropout from adj matrix
            edge_index, _ = dropout_adj(edge_index, p=0.2, force_undirected=True, training=self.training)

        if "drop_nodes" in self.augmentation and self.training:
            edge_index, _ = self.drop_nodes(edge_index)

        # compute edge weights if needed
        if self.pass_edge_weights:
            edge_weights = self.compute_edge_weights(x, edge_index)

        if self.remove_pos:
            x = x[:, 2:]  # remove positional information

        if "embedding_dropout" in self.augmentation and self.training:
            x = F.dropout(x, p=0.2, training=self.training)  # perform less aggressive dropout
            # like here: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_unet.py

        x = self.encoder(x)

        if self.pass_edge_weights:
            x = self.processor(x, edge_index, edge_weights=edge_weights)
        else:
            x = self.processor(x, edge_index)

        return self.decoder(x)
