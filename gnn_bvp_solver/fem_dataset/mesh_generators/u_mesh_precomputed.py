from dataclasses import dataclass
from functools import partial
import fenics as fn
from pyspark import SparkContext, SparkFiles
from gnn_bvp_solver.fem_dataset.mesh_generators.convert_mesh import save_msh_to_file
from gnn_bvp_solver.fem_dataset.mesh_generators.u_mesh import UMeshConfig, UMeshGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import ParametrizedMesh
import numpy as np


@dataclass
class UMeshPrecomputedConfig(UMeshConfig):
    precomputed_path: str


class UMeshPrecomputedGenerator(UMeshGenerator):
    def __init__(self, spark_context: SparkContext, randomize: bool):
        """Create mesh generator for a U mesh and precompute meshes as files"""
        super().__init__(randomize)
        self.spark_context = spark_context
        self.last_config = None

    def __call__(self) -> UMeshConfig:
        """Generate a mesh config"""
        if self.last_config is not None and (np.random.randint(16) < 15 or not self.randomize):
            return self.last_config

        config = super().__call__()  # generate config

        path = save_msh_to_file(UMeshGenerator.config2msh(config))
        self.spark_context.addFile(path)

        self.last_config = UMeshPrecomputedConfig(
            self.resolution, "UMeshPrecomputedGenerator", config.cutout_size_x, config.cutout_size_y, path
        )
        return self.last_config

    @staticmethod
    def solve_config(config: UMeshPrecomputedConfig) -> fn.Mesh:
        """Generate fenics mesh from mesh config"""
        return ParametrizedMesh(
            fn.Mesh(SparkFiles.get(config.precomputed_path[5:])),
            partial(UMeshGenerator.vis_area, config=config, is_bc=True),
            partial(UMeshGenerator.vis_area, config=config, is_bc=False),
        )
