from typing import Tuple


class CircleModule:
    def __init__(self, center: Tuple[float, float], radius: float, value: float):
        """Create a circular change for one the physical quantities

        Args:
            center (Tuple[float, float]): center of the circle
            radius (float): radius of the circle
            value (float): value to change to in this range
        """
        self.center = center
        self.radius = radius
        self.value = value

    def __call__(self, x: float, y: float) -> float:
        """Call the module to evaluate at a given position.

        Args:
            x (float): x coordinate
            y (float): y coordinate

        Returns:
            float: value of ph. quantity
        """
        x_c, y_c = self.center

        if (x - x_c) ** 2 + (y - y_c) ** 2 < self.radius:
            return self.value
