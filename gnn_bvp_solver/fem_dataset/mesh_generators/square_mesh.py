import numpy as np
from functools import partial
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import BaseMeshConfig, MeshGeneratorBase, ParametrizedMesh
from gnn_bvp_solver.fem_dataset.modules.boundary_conditions import BoundaryConditions

import fenics as fn

RES_NOISE_RANGE = [-8.0, 5.0]


class UnitSquareGenerator(MeshGeneratorBase):
    def __init__(self, resolution: int, randomize: bool = False):
        """Create mesh generator for a square mesh"""
        self.resolution = resolution
        self.randomize = randomize

    def __call__(self) -> BaseMeshConfig:
        """Generate a mesh config"""
        if self.randomize:
            res_noise = int(np.random.uniform(RES_NOISE_RANGE[0], RES_NOISE_RANGE[1]))
            return BaseMeshConfig(self.resolution + res_noise, "UnitSquareGenerator")
        else:
            return BaseMeshConfig(self.resolution, "UnitSquareGenerator")

    @staticmethod
    def vis_area(x: float, y: float) -> bool:
        """Query visible area for this mesh"""
        if x < 0 or y < 0 or x > 1.0 or y > 1.0:
            return False
        return True

    @staticmethod
    def solve_config(config: BaseMeshConfig) -> ParametrizedMesh:
        """Generate fenics mesh from mesh config"""
        return ParametrizedMesh(
            fn.UnitSquareMesh(config.resolution, config.resolution),
            BoundaryConditions("all").get_f(),
            partial(UnitSquareGenerator.vis_area),
        )
