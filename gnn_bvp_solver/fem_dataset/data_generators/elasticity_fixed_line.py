import numpy as np

from typing import Iterator
from dataclasses import dataclass
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import (
    BaseGeneratorConfig,
    GeneratorBase,
    GeneratorSolution,
    build_solution,
)
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_registry import get_mesh
from gnn_bvp_solver.fem_dataset.modules.boundary_conditions import BoundaryConditions
from gnn_bvp_solver.fem_dataset.modules.linear_elasticity_problem import ElasticityProblem
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import MeshGeneratorBase


@dataclass
class ElasticityFixedLineConfig(BaseGeneratorConfig):
    n_fix_lines_min: int
    n_fix_lines_max: int


class ElasticityFixedLineGenerator(GeneratorBase):
    def __init__(
        self, n_samples: int, mesh_generator: MeshGeneratorBase, n_fix_lines_max: int = 1, n_fix_lines_min: int = 1
    ):
        """Creates a set of linear elasticity problems.

        Args:
            n_samples (int): number of samples in dataset to generate
            mesh_generator (MeshGeneratorBase): underlying mesh generator
            n_fix_lines_max (int, optional): Max number of fixed vertical lines in the BCs. Defaults to 1.
            n_fix_lines_min (int, optional): Min number of fixed vertical lines in the BCs. Defaults to 1.
        """
        self.n_samples = n_samples
        self.n_fix_lines_min = n_fix_lines_min
        self.n_fix_lines_max = n_fix_lines_max

        super().__init__(mesh_generator)

    @staticmethod
    def solve_config(config: ElasticityFixedLineConfig, debug: bool = False) -> GeneratorSolution:
        """Solve elasticity config.

        Args:
            config (ElasticityFixedLineConfig): config to solve
            debug (bool, optional): Disable randomization for debugging. Defaults to False.

        Returns:
            GeneratorSolution: fem solution containing the mesh and visible area
        """
        parametrized_mesh = get_mesh(config.mesh_config)
        vsize = (
            config.n_fix_lines_max if debug else np.random.randint(config.n_fix_lines_min, config.n_fix_lines_max + 1)
        )
        v_lines = np.random.uniform(size=(vsize,))

        problem = ElasticityProblem(parametrized_mesh.mesh)
        problem.boundary_conditions.add_subexpression(
            BoundaryConditions("vertical_lines", x_line=v_lines, resolution=config.mesh_config.resolution).get_f()
        )

        return build_solution(problem, parametrized_mesh)

    def __iter__(self) -> Iterator[ElasticityFixedLineConfig]:
        """Generate and solve the problems.

        Returns:
           Iterator[ElasticityFixedLineConfig]: Generated solutions
        """
        for _i in range(self.n_samples):
            yield ElasticityFixedLineConfig(
                mesh_config=self.mesh_generator(),
                n_fix_lines_min=self.n_fix_lines_min,
                n_fix_lines_max=self.n_fix_lines_max,
                generator_name="ElasticityFixedLineGenerator",
            )
