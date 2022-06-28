from dataclasses import dataclass
from functools import partial
import fenics as fn
import numpy as np
from pyspark import SparkContext, SparkFiles
from gnn_bvp_solver.fem_dataset.mesh_generators.convert_mesh import save_msh_to_file
from gnn_bvp_solver.fem_dataset.mesh_generators.cylinder_mesh import CylinderGenerator, CylinderMeshConfig
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import ParametrizedMesh


@dataclass
class CylinderMeshPrecomputedConfig(CylinderMeshConfig):
    precomputed_path: str


class CylinderMeshPrecomputedGenerator(CylinderGenerator):
    def __init__(self, spark_context: SparkContext, randomize: bool):
        """Create mesh generator for a disk with hole and precompute meshes as files"""
        super().__init__(randomize)
        self.spark_context = spark_context
        self.last_config = None

    def __call__(self) -> CylinderMeshPrecomputedConfig:
        """Generate a mesh config"""
        if self.last_config is not None and (np.random.randint(16) < 15 or not self.randomize):
            return self.last_config

        config = super().__call__()  # generate config
        path = save_msh_to_file(CylinderGenerator.config2msh(config))
        self.spark_context.addFile(path)

        self.last_config = CylinderMeshPrecomputedConfig(
            self.resolution,
            "CylinderMeshPrecomputedGenerator",
            config.center_x,
            config.center_y,
            config.inner_radius,
            path,
        )
        return self.last_config

    @staticmethod
    def solve_config(config: CylinderMeshPrecomputedConfig) -> ParametrizedMesh:
        """Generate fenics mesh from mesh config"""
        return ParametrizedMesh(
            fn.Mesh(SparkFiles.get(config.precomputed_path[5:])),
            partial(CylinderGenerator.vis_area, config=config, is_bc=True),
            partial(CylinderGenerator.vis_area, config=config),
        )
