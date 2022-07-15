from dataclasses import dataclass
from functools import partial
import fenics as fn
import numpy as np
from pyspark import SparkContext, SparkFiles
from gnn_bvp_solver.fem_dataset.mesh_generators.convert_mesh import save_msh_to_file
from gnn_bvp_solver.fem_dataset.mesh_generators.disk_mesh import UnitDiskGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import BaseMeshConfig, ParametrizedMesh


@dataclass
class UnitDiskPrecomputedConfig(BaseMeshConfig):
    precomputed_path: str


class UnitDiskPrecomputedGenerator(UnitDiskGenerator):
    def __init__(self, resolution: int, spark_context: SparkContext, randomize: bool):
        """Create mesh generator for a disk and precompute meshes as files"""
        super().__init__(resolution, randomize)
        self.spark_context = spark_context
        self.last_config = None

    def __call__(self) -> UnitDiskPrecomputedConfig:
        """Generate a mesh config"""
        if self.last_config is not None and (np.random.randint(16) < 15 or not self.randomize):
            return self.last_config

        config = super().__call__()  # generate config

        path = save_msh_to_file(UnitDiskGenerator.config2msh(config))
        self.spark_context.addFile(path)

        self.last_config = UnitDiskPrecomputedConfig(self.resolution, "UnitDiskPrecomputedGenerator", path)
        return self.last_config

    @staticmethod
    def solve_config(config: UnitDiskPrecomputedConfig) -> fn.Mesh:
        """Generate fenics mesh from mesh config"""
        return ParametrizedMesh(
            fn.Mesh(SparkFiles.get(config.precomputed_path[5:])),
            partial(UnitDiskGenerator.vis_area, config=config, is_bc=True),
            partial(UnitDiskGenerator.vis_area, config=config, is_bc=False),
        )
