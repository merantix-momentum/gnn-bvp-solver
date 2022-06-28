from functools import partial
import meshio
import pygmsh
import numpy as np
from gnn_bvp_solver.fem_dataset.mesh_generators.convert_mesh import convert_msh_to_fenics
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import BaseMeshConfig, MeshGeneratorBase, ParametrizedMesh

RES_NOISE_RANGE = [-8.0, 5.0]


class UnitDiskGenerator(MeshGeneratorBase):
    def __init__(self, resolution: int, randomize: bool = False):
        """Create mesh generator for a disk"""
        self.resolution = resolution
        self.randomize = randomize

    def __call__(self) -> BaseMeshConfig:
        """Generate a mesh config"""
        if self.randomize:
            res_noise = int(np.random.uniform(RES_NOISE_RANGE[0], RES_NOISE_RANGE[1]))
            return BaseMeshConfig(self.resolution + res_noise, "UnitDiskGenerator")
        else:
            return BaseMeshConfig(self.resolution, "UnitDiskGenerator")

    @staticmethod
    def vis_area(x: float, y: float, config: BaseMeshConfig, is_bc: bool = False) -> bool:
        """Query visible area for this mesh"""
        resolution = config.resolution

        if resolution is None:
            epsilon = 0.0
        else:
            epsilon = 2.0 / resolution

        x_0 = x - 0.5
        y_0 = y - 0.5

        r_sq = np.square(x_0) + np.square(y_0)
        if r_sq > 0.25 - np.square(epsilon):
            return is_bc

        return not is_bc

    @staticmethod
    def config2msh(config: BaseMeshConfig) -> meshio.Mesh:
        """Convert mesh config to pygmsh mesh"""
        with pygmsh.occ.Geometry() as geom:
            geom.add_disk([0.5, 0.5], 0.5, mesh_size=1.0 / config.resolution)
            return geom.generate_mesh()

    @staticmethod
    def solve_config(config: BaseMeshConfig) -> ParametrizedMesh:
        """Generate fenics mesh from mesh config"""
        return ParametrizedMesh(
            convert_msh_to_fenics(UnitDiskGenerator.config2msh(config)),
            partial(UnitDiskGenerator.vis_area, config=config, is_bc=True),
            partial(UnitDiskGenerator.vis_area, config=config, is_bc=False),
        )
