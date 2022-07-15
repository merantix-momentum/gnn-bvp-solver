from dataclasses import dataclass
from typing import Iterator
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import BaseGeneratorConfig, GeneratorBase
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_registry import get_mesh
from gnn_bvp_solver.fem_dataset.modules.circle import CircleModule
from gnn_bvp_solver.fem_dataset.modules.magnetostatics_problem import MagnetostaticsProblem

import numpy as np

from gnn_bvp_solver.fem_dataset.data_generators.generator_base import GeneratorSolution, build_solution
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import MeshGeneratorBase


@dataclass
class MagneticsRandomCurrentConfig(BaseGeneratorConfig):
    n_currents_min: int
    n_currents_max: int


class MagneticsRandomCurrentGenerator(GeneratorBase):
    def __init__(
        self,
        n_samples: int,
        mesh_generator: MeshGeneratorBase,
        n_currents_max: int = 1,
        n_currents_min: int = 1,
    ):
        """Create a set of radomly distributed flowing currents of equal magitude.

        Args:
            n_samples (int): number of samples in dataset to generate
            mesh_generator (MeshGeneratorBase): underlying mesh generator
            n_currents_max (int, optional): max number of charges that will be randomly distributed. Defaults to 1.
            n_currents_min (int, optional): min number of charges that will be randomly distributed. Defaults to 1.
            resolution (int, optional): resolution of the underlying mesh. Defaults to 35.
        """
        self.n_samples = n_samples
        self.n_currents_max = n_currents_max
        self.n_currents_min = n_currents_min

        super().__init__(mesh_generator)

    @staticmethod
    def solve_config(config: MagneticsRandomCurrentConfig, debug: bool = False) -> GeneratorSolution:
        """Solve random current config."""
        parametrized_mesh = get_mesh(config.mesh_config)
        coords = parametrized_mesh.mesh.coordinates()

        problem = MagnetostaticsProblem(parametrized_mesh.mesh)
        problem.boundary_conditions.add_subexpression(parametrized_mesh.default_bc)

        range_mx = (
            config.n_currents_max if debug else np.random.randint(config.n_currents_min, config.n_currents_max + 1)
        )
        pos = GeneratorBase.distribute_on_mesh(coords, range_mx)

        for j in range(range_mx):
            problem.current_density.add_subexpression(CircleModule((pos[j, 0], pos[j, 1]), 0.005, 1.0))

        return build_solution(problem, parametrized_mesh)

    def __iter__(self) -> Iterator[MagneticsRandomCurrentConfig]:
        """Generate and solve the problems. Align n currents randomly.

        Returns:
           Iterator[MagneticsRandomCurrentConfig]: Generated solutions
        """
        for _i in range(self.n_samples):
            yield MagneticsRandomCurrentConfig(
                mesh_config=self.mesh_generator(),
                n_currents_max=self.n_currents_max,
                n_currents_min=self.n_currents_min,
                generator_name="MagneticsRandomCurrentGenerator",
            )
