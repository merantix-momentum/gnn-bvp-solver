from typing import Tuple
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import GeneratorSolution
import numpy as np
import scipy.spatial


def _min_dist_xy(p: np.array, ref_points: np.array) -> Tuple[np.array, np.array]:
    dist = scipy.spatial.distance.cdist(p, ref_points)
    closest_border = np.argmin(dist, axis=1)

    dx = ref_points[closest_border][:, 0] - p[:, 0]
    dy = ref_points[closest_border][:, 1] - p[:, 1]

    return dx, dy


def map_extend(sol: GeneratorSolution) -> GeneratorSolution:
    """Extend generator solution by adding features for distance to border and next boundary condition.

    Args:
        sol (GeneratorSolution): solution to extend

    Returns:
        GeneratorSolution: extended solution
    """
    p = np.stack([sol.quantities["x"], sol.quantities["y"]], axis=1)
    p_bc_active = sol.quantities["bdr_v0"] > 0.01

    def _apply_bc(a: np.array) -> np.array:
        return sol.mesh.default_bc(a[0], a[1])

    p_border = np.apply_along_axis(_apply_bc, axis=1, arr=p)
    bc_points = p[p_bc_active]
    border_points = p[p_border]

    sol.io_mapping["x"].append("dist_border_x")
    sol.io_mapping["x"].append("dist_border_y")
    sol.io_mapping["x"].append("dist_bc_x")
    sol.io_mapping["x"].append("dist_bc_y")
    sol.io_mapping["x"].append("dist_bc")
    sol.io_mapping["x"].append("dist_border")

    dx_border, dy_border = _min_dist_xy(p, border_points)
    sol.quantities["dist_border_x"] = dx_border
    sol.quantities["dist_border_y"] = dy_border
    sol.quantities["dist_border"] = np.sqrt(dx_border**2 + dy_border**2)

    dx_bc, dy_bc = _min_dist_xy(p, bc_points)
    sol.quantities["dist_bc_x"] = dx_bc
    sol.quantities["dist_bc_y"] = dy_bc
    sol.quantities["dist_bc"] = np.sqrt(dx_bc**2 + dy_bc**2)

    return sol
