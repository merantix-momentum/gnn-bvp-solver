from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class BaseMeshConfig:
    resolution: int
    mesh_name: str


@dataclass
class ParametrizedMesh:
    mesh: Any
    default_bc: Callable
    visible_area: Callable[[float, float], bool]


class MeshGeneratorBase(ABC):
    @abstractmethod
    def __call__(self) -> BaseMeshConfig:
        """Generate a mesh config"""
        pass

    @staticmethod
    @abstractmethod
    def solve_config(config: BaseMeshConfig) -> ParametrizedMesh:
        """Generate fenics mesh from mesh config"""
        pass
