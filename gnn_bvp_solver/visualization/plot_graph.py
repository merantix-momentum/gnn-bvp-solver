from __future__ import annotations
from typing import Any, Callable, Tuple
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import numpy as np


class Visualization:
    def __init__(
        self,
        mx: np.array,
        my: np.array,
        axis_off: bool = True,
        resolution: int = 50,
        scale: bool = True,
        figsize: Tuple[float, float] = (15, 12),
        visible_area: Callable[[float, float], bool] = None,
    ) -> None:
        """Init visualization.

        Args:
            mx (np.array): list of x coordinates of nodes in mesh
            my (np.array): list of y coordinates of nodes in mesh
            axis_off (bool, optional): Turn off axis. Defaults to True.
            resolution (int, optional): Resolution of plotting grid. Defaults to 50.
            scale (bool, optional): scale grid to data. Defaults to True.
            figsize (Tuple[float, float], optional): figsize of plot.
            visible_area (Callable[[float, float], bool], optional): area to display
        """
        self.mx = mx
        self.my = my
        self.f, self.ax = plt.subplots(figsize=figsize)
        self.c_labelsize = 45

        if scale:
            f_x = self.mx.max() - self.mx.min()
            f_y = self.my.max() - self.my.min()
            resolution_x = (f_x * resolution * 2) / (f_x + f_y)
            resolution_y = (f_y * resolution * 2) / (f_x + f_y)
        else:
            resolution_x = resolution
            resolution_y = resolution

        self.x = np.linspace(self.mx.min(), self.mx.max(), int(np.round(resolution_x)))
        self.y = np.linspace(self.my.min(), self.my.max(), int(np.round(resolution_y)))

        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.extent = (self.x.min(), self.x.max(), self.y.min(), self.y.max())

        self.vis = None
        if visible_area is not None:
            self.vis = np.zeros(self.xx.shape)
            for i in range(self.vis.shape[0]):
                for j in range(self.vis.shape[1]):
                    if visible_area(self.xx[i, j], self.yy[i, j]):
                        self.vis[i, j] = 1.0

        if axis_off:
            self.ax.axis("off")

    def plot_on_grid(
        self,
        data: np.array,
        cmap: str = "viridis",
        scatter: bool = True,
        scatter_only: bool = False,
        vmin: float = None,
        vmax: float = None,
        s: float = None,
    ) -> Visualization:
        """Plot one-dimensional data as colors on a grid.

        Args:
            data (np.array): 1d data to plot.
            cmap (str, optional): color map. Defaults to 'viridis'.
            scatter (bool, optional): plot node points. Defaults to True.
            scatter_only (bool, optional): plot only node points. Defaults to False.
            vmin (float, optional): min value for color scale. Defaults to None.
            vmax (float, optional): max value for color scale. Defaults to None.

        Returns:
            Visualization: return self
        """
        if vmax is None:
            vmax = data.max()
        if vmin is None:
            vmin = data.min()

        # interpolation
        grid_data = griddata(np.stack([self.mx, self.my], axis=1), data, (self.xx, self.yy), method="linear")
        grid_data = np.nan_to_num(grid_data)

        if scatter_only:
            self.ax.scatter(self.mx, self.my, c=data, s=25.0 if s is None else s, cmap=cmap, marker="s")

            # Change colorbar
            # scat = self.ax.scatter(self.mx, self.my, c=data, s=25.0 if s is None else s, cmap=cmap, marker="s")
            # cbar = self.f.colorbar(scat, ax = self.ax, shrink=0.7, ticks=[vmin, vmax])
            # cbar.ax.tick_params(labelsize=self.c_labelsize)

            return self

        if self.vis is not None:
            imsh = self.ax.imshow(
                grid_data,
                extent=self.extent,
                cmap=cmap,
                origin="lower",
                interpolation="bilinear",
                vmin=vmin,
                vmax=vmax,
                alpha=self.vis,
            )
            cbar = self.f.colorbar(imsh, ax=self.ax, shrink=0.7)
            cbar.ax.tick_params(labelsize=self.c_labelsize)
        else:
            imsh = self.ax.imshow(
                grid_data, extent=self.extent, cmap=cmap, origin="lower", interpolation="bilinear", vmin=vmin, vmax=vmax
            )
            cbar = self.f.colorbar(imsh, ax=self.ax, shrink=0.7)
            cbar.ax.tick_params(labelsize=self.c_labelsize)

        if scatter:
            self.ax.scatter(self.mx, self.my, s=0.5 if s is None else s, c="black")

        return self

    @staticmethod
    def get_min_max_grad_field(data_x: np.array, data_y: np.array) -> Tuple:
        """Get minimum and maximum gradient values"""
        norm = np.sqrt(data_x**2 + data_y**2)
        return norm.min(), norm.max()

    def grad_field(
        self,
        data_x: np.array,
        data_y: np.array,
        cmap: str = "viridis",
        scatter_only: bool = False,
        s: float = None,
        vmin: float = None,
        vmax: float = None,
    ) -> Visualization:
        """Plot a scalar field"""
        norm = np.sqrt(data_x**2 + data_y**2)

        if vmax is None:
            vmax = norm.max()
        if vmin is None:
            vmin = norm.min()

        grid_data = griddata(np.stack([self.mx, self.my], axis=1), norm, (self.xx, self.yy), method="linear")
        grid_data = np.nan_to_num(grid_data)

        if scatter_only:
            self.ax.scatter(self.mx, self.my, c=norm, s=25.0 if s is None else s, cmap=cmap)
            return self

        if self.vis is not None:
            imsh = self.ax.imshow(
                grid_data,
                extent=self.extent,
                cmap=cmap,
                origin="lower",
                interpolation="bilinear",
                alpha=self.vis,
                vmin=vmin,
                vmax=vmax,
            )
            cbar = self.f.colorbar(imsh, ax=self.ax, shrink=0.7)
            cbar.ax.tick_params(labelsize=self.c_labelsize)
        else:
            imsh = self.ax.imshow(
                grid_data, extent=self.extent, cmap=cmap, origin="lower", interpolation="bilinear", vmin=vmin, vmax=vmax
            )
            cbar = self.f.colorbar(imsh, ax=self.ax, shrink=0.7)
            cbar.ax.tick_params(labelsize=self.c_labelsize)

        return self

    def quiver_on_grid(
        self, data_x: np.array, data_y: np.array, normalize: bool = True, interpolate: bool = True
    ) -> Visualization:
        """Plot one-dimensional data as colors on a grid.

        Args:
            data_x (np.array): x component to plot.
            data_y (np.array): y component to plot.

        Returns:
            Visualization: return self
        """
        if normalize:
            norm = np.sqrt(data_x**2 + data_y**2)
            data_x = data_x / norm
            data_y = data_y / norm

        width = 0.005
        hal = 500.0

        if not interpolate:
            self.ax.quiver(
                self.mx,
                self.my,
                data_x,
                data_y,
                headwidth=hal,
                headaxislength=hal * 1.5,
                headlength=hal * 1.5,
                width=width,
                color="white",
            )
            return self

        # interpolation
        grid_x = griddata(np.stack([self.mx, self.my], axis=1), data_x, (self.xx, self.yy), method="linear")
        grid_y = griddata(np.stack([self.mx, self.my], axis=1), data_y, (self.xx, self.yy), method="linear")

        if self.vis is not None:
            grid_x = grid_x * self.vis
            grid_y = grid_y * self.vis

        self.ax.quiver(
            self.x,
            self.y,
            grid_x,
            grid_y,
            headwidth=hal,
            headaxislength=hal,
            headlength=hal,
            width=width,
            color="white",
        )
        return self

    def plot_line(self, idx_1: int, idx_2: int) -> None:
        """Plot a line identified by coordinate indiced of two grid points"""
        self.ax.plot([self.mx[idx_1], self.mx[idx_2]], [self.my[idx_1], self.my[idx_2]], "k-", lw=0.2)

    def plot_mesh(self, mesh: Any) -> None:
        """Plot a fenics mesh"""
        for c in mesh.cells():
            # note: we assume a triangle mesh
            self.plot_line(c[0], c[1])
            self.plot_line(c[1], c[2])
            self.plot_line(c[2], c[0])

    def show_inline(self) -> None:
        """Display the figure in an inline plot"""
        self.ax.axis("scaled")
        self.f.show()

    def to_numpy(self) -> np.array:
        """Save plot to numpy array.

        Returns:
            np.array: snapshot of the plot as numpy array
        """
        self.ax.axis("scaled")
        self.f.set_tight_layout(True)
        self.f.canvas.draw()

        result = np.frombuffer(self.f.canvas.tostring_rgb(), dtype=np.uint8)
        result = result.reshape(self.f.canvas.get_width_height()[::-1] + (3,))
        plt.close(self.f)

        return result
