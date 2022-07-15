from dataclasses import dataclass
from typing import Iterator, Tuple
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import (
    BaseGeneratorConfig,
    GeneratorBase,
    GeneratorSolution,
    build_solution,
)
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import MeshGeneratorBase
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_registry import get_mesh
from gnn_bvp_solver.fem_dataset.modules.circle import CircleModule
from gnn_bvp_solver.fem_dataset.modules.plate import PlateModule
from gnn_bvp_solver.fem_dataset.modules.electrostatics_problem import ElectrostaticsProblem

import numpy as np


@dataclass
class CondenserPlatesConfig(BaseGeneratorConfig):
    height: float
    distance: float
    theta: float
    conductor: bool


class CondenserPlatesGenerator(GeneratorBase):
    def __init__(
        self,
        n_samples: int,
        mesh_generator: MeshGeneratorBase,
        height_range: Tuple[float, float],
        distance_range: Tuple[float, float],
        n_rotations: int = 1,
        conductor: bool = False,
    ):
        """Creates a set of condenser plates and optionally a conductor in between.

        Args:
            n_samples (int): number of samples in dataset to generate
            mesh_generator (MeshGeneratorBase): underlying mesh generator
            height_range (Tuple): min and max height
            distance_range (Tuple): min and max distance
            n_rotations (int): number of rotations
            conductor (bool, optional): Include a conductor between the plates. Defaults to False.
        """
        self.n_rotations = n_rotations
        self.height_range = height_range
        self.distance_range = distance_range
        self.conductor = conductor
        self.n_samples = n_samples // n_rotations

        super().__init__(mesh_generator)

    @staticmethod
    def solve_config(config: CondenserPlatesConfig) -> GeneratorSolution:
        """Solve condenser config.

        Args:
            config (CondenserPlatesConfig): config to solve

        Returns:
            GeneratorSolution: fem solution containing the mesh and visible area
        """
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

        if config.conductor:
            problem.permittivity.add_subexpression(CircleModule((0.5, 0.5), 0.005, 1000000.0))

        return build_solution(problem, parametrized_mesh)

    def __iter__(self) -> Iterator[CondenserPlatesConfig]:
        """Generate and solve the problems rotating condenser plates and varying height and distance.

        Returns:
           Iterator[CondenserPlatesConfig]: Generated solutions
        """
        samples_height = int(np.sqrt(self.n_samples))
        samples_distance = self.n_samples // samples_height

        for height in np.linspace(self.height_range[0], self.height_range[1], num=samples_height):
            for distance in np.linspace(self.distance_range[0], self.distance_range[1], num=samples_distance):
                for theta in np.linspace(0, 2 * np.pi, num=self.n_rotations, endpoint=False):
                    yield CondenserPlatesConfig(
                        mesh_config=self.mesh_generator(),
                        height=height,
                        distance=distance,
                        theta=theta,
                        conductor=self.conductor,
                        generator_name="CondenserPlatesGenerator",
                    )
