from dataclasses import dataclass
from functools import partial
import fenics as fn
from pyspark import SparkContext, SparkFiles
from gnn_bvp_solver.fem_dataset.mesh_generators.convert_mesh import save_msh_to_file
from gnn_bvp_solver.fem_dataset.mesh_generators.l_mesh import LMeshConfig, LMeshGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import ParametrizedMesh
import numpy as np


@dataclass
class LMeshPrecomputedConfig(LMeshConfig):
    precomputed_path: str


class LMeshPrecomputedGenerator(LMeshGenerator):
    def __init__(self, spark_context: SparkContext, randomize: bool):
        """Create mesh generator for an L mesh"""
        super().__init__(randomize)
        self.spark_context = spark_context
        self.last_config = None

    def __call__(self) -> LMeshPrecomputedConfig:
        """Generate a mesh config"""
        if self.last_config is not None and (np.random.randint(16) < 15 or not self.randomize):
            return self.last_config

        config = super().__call__()  # generate config

        path = save_msh_to_file(LMeshGenerator.config2msh(config))
        self.spark_context.addFile(path)

        self.last_config = LMeshPrecomputedConfig(
            self.resolution,
            "LMeshPrecomputedGenerator",
            config.cutout_location,
            config.cutout_size_x,
            config.cutout_size_y,
            path,
        )
        return self.last_config

    @staticmethod
    def solve_config(config: LMeshPrecomputedConfig) -> fn.Mesh:
        """Generate fenics mesh from mesh config"""
        return ParametrizedMesh(
            fn.Mesh(SparkFiles.get(config.precomputed_path[5:])),
            partial(LMeshGenerator.vis_area, config=config, is_bc=True),
            partial(LMeshGenerator.vis_area, config=config, is_bc=False),
        )
