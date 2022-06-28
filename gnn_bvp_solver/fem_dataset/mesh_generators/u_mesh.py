from dataclasses import dataclass
from functools import partial
import fenics as fn
import meshio
import pygmsh
import numpy as np
from gnn_bvp_solver.fem_dataset.mesh_generators.convert_mesh import convert_msh_to_fenics
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import BaseMeshConfig, MeshGeneratorBase, ParametrizedMesh


@dataclass
class UMeshConfig(BaseMeshConfig):
    cutout_size_x: float
    cutout_size_y: float


UM_CONST_RES = 9
CUTOUT_RANGE_X = [0.2, 0.6]
CUTOUT_RANGE_Y = [0.2, 0.8]


class UMeshGenerator(MeshGeneratorBase):
    def __init__(self, randomize: bool = False):
        """Create mesh generator for a U mesh"""
        # fixed resolution
        self.resolution = UM_CONST_RES
        self.randomize = randomize

    def __call__(self) -> UMeshConfig:
        """Generate a mesh config"""
        if self.randomize:
            cutout_size_x = np.random.uniform(CUTOUT_RANGE_X[0], CUTOUT_RANGE_X[1])
            cutout_size_y = np.random.uniform(CUTOUT_RANGE_Y[0], CUTOUT_RANGE_Y[1])
        else:
            cutout_size_x = 0.5
            cutout_size_y = 0.75

        return UMeshConfig(self.resolution, "UMeshGenerator", cutout_size_x, cutout_size_y)

    @staticmethod
    def vis_area(x: float, y: float, config: UMeshConfig, is_bc: bool = False) -> bool:
        """Query visible area for this mesh"""
        cutout_size_x = config.cutout_size_x
        cutout_size_y = config.cutout_size_y

        epsilon = 0.001 if is_bc else 0.0

        # outside bounding box
        if x < 0 + epsilon or y < 0 + epsilon or x > 1.0 - epsilon or y > 1.0 - epsilon:
            return is_bc

        # inside cutout
        cond_x = (x > (0.5 - cutout_size_x / 2.0) - epsilon) and (x < (0.5 + cutout_size_x / 2.0) + epsilon)
        cond_y = y > 1.0 - cutout_size_y - epsilon
        if cond_x and cond_y:
            return is_bc

        return not is_bc

    @staticmethod
    def config2msh(config: UMeshConfig) -> meshio.Mesh:
        """Convert mesh config to pygmsh mesh"""
        with pygmsh.occ.Geometry() as geom:
            cutout_size_x = config.cutout_size_x
            cutout_size_y = config.cutout_size_y

            g1 = geom.add_rectangle([0.0, 0.0, 0.0], 1.0, 1.0)
            g2 = geom.add_rectangle([0.5 - cutout_size_x / 2.0, 1.0 - cutout_size_y, 0.0], cutout_size_x, cutout_size_y)

            geom.boolean_difference(g1, g2)
            return geom.generate_mesh()

    @staticmethod
    def solve_config(config: UMeshConfig) -> fn.Mesh:
        """Generate fenics mesh from mesh config"""
        mesh = UMeshGenerator.config2msh(config)
        return ParametrizedMesh(
            convert_msh_to_fenics(mesh),
            partial(UMeshGenerator.vis_area, config=config, is_bc=True),
            partial(UMeshGenerator.vis_area, config=config, is_bc=False),
        )
