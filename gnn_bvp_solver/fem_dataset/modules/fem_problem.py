from abc import ABC, abstractmethod
from typing import Dict, Iterable, Tuple
from gnn_bvp_solver.fem_dataset.recursive_user_expression import RecursiveUserExpression

import fenics as fn
import numpy as np


class FemProblem(ABC):
    def __init__(self, mesh: fn.Mesh):
        """Init FEM Problem with given mesh.

        Args:
            mesh: FEM mesh to evaluate on.
        """
        self.boundary_conditions = RecursiveUserExpression(False)
        self.mesh = mesh

    @abstractmethod
    def solve(self) -> Dict[str, np.array]:
        """Using fenics to find the solution for an fem problem

        Returns:
            Dict[str, np.array]: solution
        """
        pass

    @staticmethod
    @abstractmethod
    def input_output_mapping() -> Dict[str, Iterable[str]]:
        """Specify which quantities are input or output.

        Returns:
            Dict[str, Iterable[str]]: input / output values of this problem
        """
        pass

    @staticmethod
    @abstractmethod
    def physical_quantities() -> Iterable[Tuple]:
        """Return physical quantities present in this simulation.

        Returns:
            Iterable: all quantities (can be 1d or nd)
        """
        pass
