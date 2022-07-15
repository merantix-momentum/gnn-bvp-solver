from dataclasses import dataclass
from typing import Iterator
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import BaseGeneratorConfig, GeneratorBase, GeneratorSolution
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import MeshGeneratorBase
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_registry import get_mesh
from gnn_bvp_solver.fem_dataset.modules.electrostatics_problem import ElectrostaticsProblem
from gnn_bvp_solver.fem_dataset.modules.circle import CircleModule
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import build_solution

import numpy as np


@dataclass
class ElectricsRandomChargeConfig(BaseGeneratorConfig):
    n_charges_min: int
    n_charges_max: int


class ElectricsRandomChargeGenerator(GeneratorBase):
    def __init__(
        self,
        n_samples: int,
        mesh_generator: MeshGeneratorBase,
        n_charges_max: int = 1,
        n_charges_min: int = 1,
    ):
        """Create a set of randomly distributed charges with equal density.

        Args:
            n_samples (int): number of samples in dataset to generate
            mesh_generator (MeshGeneratorBase): underlying mesh generator
            n_charges_max (int, optional): max number of charges that will be randomly distributed. Defaults to 1.
            n_charges_min (int, optional): min number of charges that will be randomly distributed. Defaults to 1.
        """
        self.n_samples = n_samples
        self.n_charges_max = n_charges_max
        self.n_charges_min = n_charges_min

        super().__init__(mesh_generator)

    @staticmethod
    def solve_config(config: ElectricsRandomChargeConfig, debug: bool = False) -> GeneratorSolution:
        """Solve random charge config.

        Args:
            config (ElectricsRandomChargeConfig): config to solve
            debug (bool, optional): Disable randomization for debugging. Defaults to False.

        Returns:
            GeneratorSolution: fem solution containing the mesh and visible area
        """
        parametrized_mesh = get_mesh(config.mesh_config)
        coords = parametrized_mesh.mesh.coordinates()

        problem = ElectrostaticsProblem(parametrized_mesh.mesh)
        problem.boundary_conditions.add_subexpression(parametrized_mesh.default_bc)
        range_mx = config.n_charges_max if debug else np.random.randint(config.n_charges_min, config.n_charges_max + 1)
        pos = GeneratorBase.distribute_on_mesh(coords, range_mx)

        for j in range(range_mx):
            problem.charge_density.add_subexpression(CircleModule((pos[j, 0], pos[j, 1]), 0.005, 1.0))

        return build_solution(problem, parametrized_mesh)

    def __iter__(self) -> Iterator[ElectricsRandomChargeConfig]:
        """Generate and solve the problems. Align n currents randomly.

        Returns:
           Iterator[ElectricsRandomChargeConfig]: Generated solutions
        """
        for _i in range(self.n_samples):
            yield ElectricsRandomChargeConfig(
                mesh_config=self.mesh_generator(),
                n_charges_min=self.n_charges_min,
                n_charges_max=self.n_charges_max,
                generator_name="ElectricsRandomChargeGenerator",
            )
