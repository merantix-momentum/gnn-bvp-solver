from dataclasses import dataclass
from functools import partial
import meshio
import pygmsh
import numpy as np
from gnn_bvp_solver.fem_dataset.mesh_generators.convert_mesh import convert_msh_to_fenics
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import BaseMeshConfig, MeshGeneratorBase, ParametrizedMesh


@dataclass
class LMeshConfig(BaseMeshConfig):
    cutout_location: int
    cutout_size_x: float
    cutout_size_y: float


LM_CONST_RES = 9
CUTOUT_RANGE = [0.2, 0.8]


class LMeshGenerator(MeshGeneratorBase):
    def __init__(self, randomize: bool = False):
        """Create mesh generator for an L mesh and precompute meshes as files"""
        # fixed resolution
        self.resolution = LM_CONST_RES
        self.randomize = randomize

    def __call__(self) -> LMeshConfig:
        """Generate a mesh config"""
        if self.randomize:
            cutout_location = np.random.randint(4)
            cutout_size_x = np.random.uniform(CUTOUT_RANGE[0], CUTOUT_RANGE[1])
            cutout_size_y = np.random.uniform(CUTOUT_RANGE[0], CUTOUT_RANGE[1])
        else:
            cutout_location = 0
            cutout_size_x = 0.5
            cutout_size_y = 0.5

        return LMeshConfig(self.resolution, "LMeshGenerator", cutout_location, cutout_size_x, cutout_size_y)

    @staticmethod
    def vis_area(x: float, y: float, config: LMeshConfig, is_bc: bool = False) -> bool:
        """Query visible area for this mesh"""
        cutout_size_x = config.cutout_size_x
        cutout_size_y = config.cutout_size_y
        cutout_location = config.cutout_location

        epsilon = 0.001 if is_bc else 0.0

        # outside bounding box
        if x < 0 + epsilon or y < 0 + epsilon or x > 1.0 - epsilon or y > 1.0 - epsilon:
            return is_bc

        if cutout_location == 0:
            if x < cutout_size_x + epsilon and y < cutout_size_y + epsilon:
                return is_bc
        elif cutout_location == 1:
            if x > (1.0 - cutout_size_x) - epsilon and y < cutout_size_y + epsilon:
                return is_bc
        elif cutout_location == 2:
            if x < cutout_size_x + epsilon and y > (1 - cutout_size_y) - epsilon:
                return is_bc
        else:
            if x > (1.0 - cutout_size_x) - epsilon and y > (1 - cutout_size_y) - epsilon:
                return is_bc

        return not is_bc

    @staticmethod
    def config2msh(config: LMeshConfig) -> meshio.Mesh:
        """Convert mesh config to pygmsh mesh"""
        with pygmsh.occ.Geometry() as geom:
            cutout_size_x = config.cutout_size_x
            cutout_size_y = config.cutout_size_y
            cutout_location = config.cutout_location

            # mesh_size = 1.25 / config.resolution
            # ignore resolution

            g1 = geom.add_rectangle([0.0, 0.0, 0.0], 1.0, 1.0)

            diff_x = 0.5 - cutout_size_x
            diff_y = 0.5 - cutout_size_y

            if cutout_location == 0:
                g2 = geom.add_rectangle([0.0, 0.0, 0.0], cutout_size_x, cutout_size_y)
            elif cutout_location == 1:
                g2 = geom.add_rectangle([0.5 + diff_x, 0.0, 0.0], cutout_size_x, cutout_size_y)
            elif cutout_location == 2:
                g2 = geom.add_rectangle([0.0, 0.5 + diff_y, 0.0], cutout_size_x, cutout_size_y)
            else:
                g2 = geom.add_rectangle([0.5 + diff_x, 0.5 + diff_y, 0.0], cutout_size_x, cutout_size_y)

            geom.boolean_difference(g1, g2)
            return geom.generate_mesh()

    @staticmethod
    def solve_config(config: LMeshConfig) -> ParametrizedMesh:
        """Generate fenics mesh from mesh config"""
        mesh = LMeshGenerator.config2msh(config)
        return ParametrizedMesh(
            convert_msh_to_fenics(mesh),
            partial(LMeshGenerator.vis_area, config=config, is_bc=True),
            partial(LMeshGenerator.vis_area, config=config, is_bc=False),
        )
