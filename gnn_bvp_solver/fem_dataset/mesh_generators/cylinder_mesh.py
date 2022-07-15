from dataclasses import dataclass
from functools import partial
import meshio
import pygmsh
import numpy as np
from gnn_bvp_solver.fem_dataset.mesh_generators.convert_mesh import convert_msh_to_fenics
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import BaseMeshConfig, MeshGeneratorBase, ParametrizedMesh


CM_CONST_RES = 8
CM_CONST_INNER_RADIUS_RANGE = [0.05, 0.25]
CM_CONST_CENTER_X_RANGE = [0.35, 0.65]
CM_CONST_CENTER_Y_RANGE = [0.35, 0.65]


@dataclass
class CylinderMeshConfig(BaseMeshConfig):
    center_x: float
    center_y: float
    inner_radius: float


class CylinderGenerator(MeshGeneratorBase):
    def __init__(self, randomize: bool = False):
        """Create mesh generator for a disk with hole"""
        # fixed resolution
        self.resolution = CM_CONST_RES
        self.randomize = randomize

    def __call__(self) -> CylinderMeshConfig:
        """Generate a mesh config"""
        if self.randomize:
            inner_radius = np.random.uniform(CM_CONST_INNER_RADIUS_RANGE[0], CM_CONST_INNER_RADIUS_RANGE[1])
            center_x = np.random.uniform(CM_CONST_CENTER_X_RANGE[0], CM_CONST_CENTER_X_RANGE[1])
            center_y = np.random.uniform(CM_CONST_CENTER_Y_RANGE[0], CM_CONST_CENTER_Y_RANGE[1])
        else:
            inner_radius = 0.12
            center_x = 0.5
            center_y = 0.5

        return CylinderMeshConfig(self.resolution, "CylinderGenerator", center_x, center_y, inner_radius)

    @staticmethod
    def vis_area(x: float, y: float, config: CylinderMeshConfig, is_bc: bool = False) -> bool:
        """Query visible area for this mesh"""
        resolution = config.resolution
        center_x = config.center_x
        center_y = config.center_y
        inner_radius = config.inner_radius

        epsilon = 1.0 / resolution if is_bc else 0.0

        x_0 = x - center_x
        y_0 = y - center_y

        r_sq = np.square(x_0) + np.square(y_0)
        r_sq_outer = np.square(x - 0.5) + np.square(y - 0.5)

        if r_sq_outer > 0.25 - np.square(epsilon):
            return is_bc
        if r_sq < np.square(inner_radius) + np.square(epsilon) * 0.5:
            return is_bc

        return not is_bc

    @staticmethod
    def config2msh(config: CylinderMeshConfig) -> meshio.Mesh:
        """Convert mesh config to pygmsh mesh"""
        center_x = config.center_x
        center_y = config.center_y
        inner_radius = config.inner_radius

        with pygmsh.occ.Geometry() as geom:
            g1 = geom.add_disk([0.5, 0.5], 0.5)
            g2 = geom.add_disk([center_x, center_y], inner_radius)
            geom.boolean_difference(g1, g2)
            return geom.generate_mesh()

    @staticmethod
    def solve_config(config: CylinderMeshConfig) -> ParametrizedMesh:
        """Generate fenics mesh from mesh config"""
        return ParametrizedMesh(
            convert_msh_to_fenics(CylinderGenerator.config2msh(config)),
            partial(CylinderGenerator.vis_area, config=config, is_bc=True),
            partial(CylinderGenerator.vis_area, config=config),
        )
