import numpy as np


class PlateModule:
    def __init__(self, distance: float, height: float, value: float, rotation: float, resolution: int):
        """Create a change in form of a thin plate for one the physical quantities

        Args:
            distance (float): distance from center
            height (float): height of plat
            value (float): value of ph. quantity to set
            rotation (float): rotation of plate
            resolution (int): resolution of mesh, needed to avoid rounding errors
        """
        self.distance = distance
        self.height = height
        self.value = value
        self.resolution = resolution
        self.rotation = rotation

    def __call__(self, x: float, y: float):
        """Call the module to evaluate at a given position.

        Args:
            x (float): x coordinate
            y (float): y coordinate

        Returns:
            float: value of ph. quantity
        """
        dx = (x - 0.5) * np.cos(self.rotation) - (y - 0.5) * np.sin(self.rotation) + 0.5
        dy = (x - 0.5) * np.sin(self.rotation) + (y - 0.5) * np.cos(self.rotation) + 0.5

        if dy > (0.5 + (self.height / 2.0)) or dy < (0.5 - self.height / 2.0):
            return

        # divide by resolution to avoid disappearance if no nodes are hit
        if np.abs((dx - 0.5) - self.distance) < 1.0 / self.resolution:
            return self.value
