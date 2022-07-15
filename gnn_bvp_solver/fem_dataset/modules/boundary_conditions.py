from typing import Callable, Iterable
import numpy as np


class BoundaryConditions:
    def __init__(self, name: str, **kwargs):
        """Define different types of boundary conditions"""
        self.name = name
        self.kwargs = kwargs

    def get_f(self) -> Callable:
        """Get predefined boundary conditions (mostly for square mesh)"""
        if self.name == "all":
            return self.all_boundaries
        elif self.name == "left_and_right":
            return self.left_and_right
        elif self.name == "left":
            return self.left
        elif self.name == "right":
            return self.right
        elif self.name == "top_and_bottom":
            return self.top_and_bottom
        elif self.name == "vertical_lines":
            return self.f_vertical_lines(**self.kwargs)
        else:
            raise ValueError(f"name {self.name} not found")

    @staticmethod
    def all_boundaries(x: float, y: float) -> bool:
        """Set all boundaries of a square mesh to true.

        Args:
            x (float): x coordinate
            y (float): y coordinate

        Returns:
            bool: whether boundary is active at this point
        """
        return x < 0.00001 or x > 0.99999 or y < 0.00001 or y > 0.99999

    @staticmethod
    def left_and_right(x: float, y: float) -> bool:
        """Set left and right boundaries of a square mesh to true.

        Args:
            x (float): x coordinate
            y (float): y coordinate

        Returns:
            bool: whether boundary is active at this point
        """
        return x < 0.00001 or x > 0.99999

    @staticmethod
    def left(x: float, y: float) -> bool:
        """Set left boundaries of a square mesh to true.

        Args:
            x (float): x coordinate
            y (float): y coordinate

        Returns:
            bool: whether boundary is active at this point
        """
        return x < 0.00001

    @staticmethod
    def right(x: float, y: float, bdr: bool) -> bool:
        """Set right boundaries of a square mesh to true.

        Args:
            x (float): x coordinate
            y (float): y coordinate

        Returns:
            bool: whether boundary is active at this point
        """
        return x > 0.99999

    @staticmethod
    def top_and_bottom(x: float, y: float) -> bool:
        """Set top and bottom boundaries of a square mesh to true.

        Args:
            x (float): x coordinate
            y (float): y coordinate

        Returns:
            bool: whether boundary is active at this point
        """
        return y < 0.00001 or y > 0.99999

    @staticmethod
    def f_vertical_lines(x_line: Iterable[float], resolution: int) -> Callable[[float, float], bool]:
        """Create boundary condition as a vertical line.

        Args:
            x_line (float): x_line coordinate of line
            resolution (int): resolution of mesh, needed to avoid rounding errors

        Returns:
            bool: whether boundary is active at this point
        """
        x_line = np.array(x_line)
        return lambda x, y: np.min(np.abs(x - x_line)) < 0.5 / resolution
