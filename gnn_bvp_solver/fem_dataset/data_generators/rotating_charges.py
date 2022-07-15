from dataclasses import dataclass
from typing import Iterator, Tuple
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import BaseGeneratorConfig, GeneratorBase
from gnn_bvp_solver.fem_dataset.modules.electrostatics_problem import ElectrostaticsProblem
from gnn_bvp_solver.fem_dataset.modules.circle import CircleModule
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import solution_type, i_o_mapping_type

import numpy as np
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import build_solution
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import MeshGeneratorBase

from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_registry import get_mesh


@dataclass
class RotatingChargesConfig(BaseGeneratorConfig):
    theta: float
    distance: float
    n_charges: int


class RotatingChargesGenerator(GeneratorBase):
    def __init__(
        self,
        n_samples: int,
        mesh_generator: MeshGeneratorBase,
        n_charges: int = 1,
        distance_r: float = 0.1,
    ):
        """Creates a set of rotating charges with equal charge

        Args:
            n_samples (int, optional): number of rotating steps.
            mesh_generator (MeshGeneratorBase): underlying mesh generator
            n_charges (int, optional): number of rotating charges. Defaults to 1.
            distance_r (float, optional): radial distance to center. Defaults to 0.1.
        """
        self.n_charges = n_charges
        self.n_samples = n_samples
        self.distance = distance_r / np.sqrt(2.0)

        super().__init__(mesh_generator)

    @staticmethod
    def solve_config(config: RotatingChargesConfig) -> Tuple[solution_type, i_o_mapping_type]:
        """Solve config combining plates and charges."""
        parametrized_mesh = get_mesh(config.mesh_config)

        problem = ElectrostaticsProblem(parametrized_mesh.mesh)
        problem.boundary_conditions.add_subexpression(parametrized_mesh.default_bc)

        for i in range(config.n_charges):
            theta_charge = config.theta + (i * 2.0 * np.pi) / config.n_charges

            dx = config.distance * np.cos(theta_charge) - config.distance * np.sin(theta_charge)
            dy = config.distance * np.sin(theta_charge) + config.distance * np.cos(theta_charge)

            problem.charge_density.add_subexpression(CircleModule((0.5 + dx, 0.5 + dy), 0.005, 1.0))

        return build_solution(problem, parametrized_mesh)

    def __iter__(self) -> Iterator[float]:
        """Generate and solve the problems. Align n charges on a circle and rotate.

        Returns:
           Iterator[Tuple[solution_type, i_o_mapping_type]]: Generated solutions
        """
        for theta in np.linspace(0, 2 * np.pi, num=self.n_samples, endpoint=False):
            yield RotatingChargesConfig(
                theta=theta,
                distance=self.distance,
                n_charges=self.n_charges,
                mesh_config=self.mesh_generator(),
                generator_name="RotatingChargesGenerator",
            )
