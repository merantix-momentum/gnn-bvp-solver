from dataclasses import dataclass
from typing import Iterator, Tuple
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import BaseGeneratorConfig, GeneratorBase
from gnn_bvp_solver.fem_dataset.modules import boundary_conditions
from gnn_bvp_solver.fem_dataset.modules.electrostatics_problem import ElectrostaticsProblem
from gnn_bvp_solver.fem_dataset.modules.plate import PlateModule
from gnn_bvp_solver.fem_dataset.modules.circle import CircleModule

import numpy as np
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import GeneratorSolution, build_solution
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import MeshGeneratorBase

from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_registry import get_mesh


@dataclass
class PlatesAndChargesConfig(BaseGeneratorConfig):
    distance: float
    height: float
    theta: float


class PlatesAndChargesGenerator(GeneratorBase):
    def __init__(
        self,
        n_samples: int,
        mesh_generator: MeshGeneratorBase,
        height_range: Tuple[float, float],
        distance_range: Tuple[float, float],
        n_rotations: int = 1,
    ):
        """Creates a set of condenser plates and additional point charges.

        Args:
            n_samples (int): number of samples in dataset to generate
            mesh_generator (MeshGeneratorBase): underlying mesh generator
            height_range (Tuple): min and max height
            distance_range (Tuple): min and max distance
            n_rotations (int, optional): Rotations of the condenser plates. Defaults to 1.
        """
        self.n_rotations = n_rotations
        self.height_range = height_range
        self.distance_range = distance_range
        self.n_samples = n_samples // n_rotations
        self.boundary_conditions = boundary_conditions

        super().__init__(mesh_generator)

    @staticmethod
    def solve_config(config: PlatesAndChargesConfig) -> GeneratorSolution:
        """Solve config combining plates and charges."""
        parametrized_mesh = get_mesh(config.mesh_config)

        problem = ElectrostaticsProblem(parametrized_mesh.mesh)
        problem.boundary_conditions.add_subexpression(parametrized_mesh.default_bc)
        problem.charge_density.add_subexpression(
            PlateModule(
                config.distance, config.height, 1.0, rotation=config.theta, resolution=config.mesh_config.resolution
            )
        )
        problem.charge_density.add_subexpression(
            PlateModule(
                -config.distance, config.height, -1.0, rotation=config.theta, resolution=config.mesh_config.resolution
            )
        )

        problem.charge_density.add_subexpression(
            CircleModule((0.5, 0.5 + max(config.height, config.distance) + 0.1), 0.005, -1.0)
        )
        problem.charge_density.add_subexpression(
            CircleModule((0.5, 0.5 - max(config.height, config.distance) - 0.1), 0.005, 1.0)
        )

        return build_solution(problem, parametrized_mesh)

    def __iter__(self) -> Iterator[PlatesAndChargesConfig]:
        """Generate and solve the problems.

        Returns:
           Iterator[PlatesAndChargesConfig]: Generated solutions
        """
        samples_height = int(np.sqrt(self.n_samples))
        samples_distance = self.n_samples // samples_height

        for height in np.linspace(self.height_range[0], self.height_range[1], num=samples_height):
            for distance in np.linspace(self.distance_range[0], self.distance_range[1], num=samples_distance):
                for theta in np.linspace(0, 2 * np.pi, num=self.n_rotations, endpoint=False):
                    yield PlatesAndChargesConfig(
                        distance=distance,
                        height=height,
                        theta=theta,
                        generator_name="PlatesAndChargesGenerator",
                        mesh_config=self.mesh_generator(),
                    )
