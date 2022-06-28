import torch
from torch.nn import ReLU, Sequential, Linear, Identity


class TwoLayerMLP(torch.nn.Module):
    def __init__(self, n_input: int = 128, n_hidden: int = 128, n_output: int = 128, is_output: bool = False):
        """Simple MLP with two linear layers and relu nonlinearity"""
        super().__init__()

        self.layers = Sequential(
            Linear(n_input, n_hidden),
            ReLU(),
            Linear(n_hidden, n_output),
            Identity() if is_output else ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward"""
        return self.layers(x)
