from abc import ABC, abstractmethod
from dataclasses import dataclass
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import MeshGeneratorBase
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import BaseMeshConfig, ParametrizedMesh
from gnn_bvp_solver.fem_dataset.modules.fem_problem import FemProblem
from typing import Iterable, Iterator, Dict

import numpy as np


i_o_mapping_type = Dict[str, Iterable[str]]
solution_type = Dict[str, np.array]


@dataclass
class BaseGeneratorConfig:
    mesh_config: BaseMeshConfig
    generator_name: str


@dataclass
class GeneratorSolution:
    quantities: Dict
    io_mapping: Dict
    visible_area: np.array
    mesh: ParametrizedMesh


def build_solution(problem: FemProblem, parametrized_mesh: ParametrizedMesh) -> GeneratorSolution:
    """Construct a generator solution by solving an fem problem on a mesh.

    Args:
        problem (FemProblem): fem problem to solve
        parametrized_mesh (ParametrizedMesh): mesh info to be stored for future reference

    Returns:
        GeneratorSolution: generated solution
    """
    return GeneratorSolution(
        problem.solve(), problem.input_output_mapping(), parametrized_mesh.visible_area, parametrized_mesh
    )


class GeneratorBase(ABC):
    def __init__(self, mesh_generator: MeshGeneratorBase):
        """Creates the base class for all fem generators.

        Args:
            mesh_generator (MeshGeneratorBase): each fem generator needs access to a mesh generator
        """
        self.mesh_generator = mesh_generator

    @abstractmethod
    def __iter__(self) -> Iterator[BaseGeneratorConfig]:
        """Generate fem configs that can be turned into problems and solved.

        Returns:
           Iterator[BaseGeneratorConfig]: Generated configs
        """
        pass

    @staticmethod
    def distribute_on_mesh(coords: np.array, n_samples: int) -> np.array:
        """Randomly select a range of coordinates without replacement.

        Args:
            coords (np.array): coordinates to sample from
            n_samples (int): number of coordinates to sample

        Returns:
            np.array: sampled coordinates
        """
        return coords[np.random.choice(coords.shape[0], n_samples, replace=False), :]

    @staticmethod
    @abstractmethod
    def solve_config(config: BaseGeneratorConfig) -> GeneratorSolution:
        """Generate an fem problem for a given config and solve it.

        Args:
            config (BaseGeneratorConfig): config to solve

        Yields:
            GeneratorSolution: fem solution containing the mesh and visible area
        """
        pass
