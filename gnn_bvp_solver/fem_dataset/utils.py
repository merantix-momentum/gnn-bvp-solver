from typing import Iterable
import torch
import numpy as np
import fenics


def stack_data(data: Iterable[np.array]) -> torch.tensor:
    """Helper function for converting data to pytorch and stacking it

    Args:
        data (Iterable[np.array]): List of numpy arrays

    Returns:
        torch.Tensor: Aggregated data
    """
    data = np.stack([d.astype(np.float32).flatten() for d in data], axis=1)
    return torch.tensor(data, dtype=torch.float)


def extract_edges_from_triangle_mesh(mesh: fenics.Mesh) -> np.array:
    """Extract an unfiltered list of edges from a triangle mesh.

    Args:
        mesh (fenics.Mesh): The mesh to iterate over.

    Returns:
        np.array: Array containing list of edges indicated by indexes.
    """
    edges_source = []
    edges_sink = []
    for c in mesh.cells():
        # note: we assume a triangle mesh
        edges_source += [c[0], c[1], c[2], c[1], c[2], c[0]]
        edges_sink += [c[1], c[2], c[0], c[0], c[1], c[2]]

    # note that this array is still unfiltered and might contain duplicates
    # we have a convenient way to filter with pt geometric
    return np.array([edges_source, edges_sink], dtype=np.int64)
