from typing import List, Union
from gnn_bvp_solver.fem_dataset.data_generators.elasticity_fixed_line import ElasticityFixedLineGenerator
from gnn_bvp_solver.fem_dataset.data_generators.electrics_random_charge import ElectricsRandomChargeGenerator
from gnn_bvp_solver.fem_dataset.data_generators.magnetics_random_current import MagneticsRandomCurrentGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.square_mesh import UnitSquareGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.cylinder_mesh import CylinderGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.l_mesh import LMeshGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.disk_mesh import UnitDiskGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.u_mesh import UMeshGenerator
from gnn_bvp_solver.visualization.plot_graph import Visualization
from gnn_bvp_solver.fem_dataset.data_generators.extend_solution import map_extend

import matplotlib.pyplot as plt
import wandb
from matplotlib.colors import ListedColormap


def plot_quantities_multimesh(
    plot: Union[None, List] = None, randomize: bool = False, fem_g: int = 2, show: bool = True
) -> None:
    """Visualize quantities for multiple meshes"""
    if plot is None:
        plot = ["bdr", "g_field", "q", "bdr_dist"]

    if fem_g == 2:
        es = ElectricsRandomChargeGenerator
        em = MagneticsRandomCurrentGenerator

        fem_g = [es, em]
    elif fem_g == 1:
        fem_g = [ElectricsRandomChargeGenerator]

    g_small_square = UnitSquareGenerator(15, randomize)
    g_cylinder = CylinderGenerator(randomize)
    g_disk = UnitDiskGenerator(15, randomize)
    g_lmesh = LMeshGenerator(randomize)
    g_umesh = UMeshGenerator(randomize)

    if randomize:
        mesh_g = [g_small_square, g_cylinder, g_disk, g_lmesh]  # g_umesh
    else:
        mesh_g = [g_small_square, g_cylinder, g_disk, g_lmesh, g_umesh]  # g_umesh

    if "bdr" in plot:
        f1, ax1 = plt.subplots(len(fem_g), len(mesh_g), figsize=(len(mesh_g) * 5, len(fem_g) * 5), squeeze=False)
        f1.set_tight_layout(True)

    if "q" in plot:
        f, ax = plt.subplots(len(fem_g), len(mesh_g), figsize=(len(mesh_g) * 5, len(fem_g) * 5), squeeze=False)
        f.set_tight_layout(True)

    if "g_field" in plot:
        f2, ax2 = plt.subplots(len(fem_g), len(mesh_g), figsize=(len(mesh_g) * 5, len(fem_g) * 5), squeeze=False)
        f2.set_tight_layout(True)

    if "bdr_dist" in plot:
        f3, ax3 = plt.subplots(len(fem_g), len(mesh_g), figsize=(len(mesh_g) * 5, len(fem_g) * 5), squeeze=False)
        f4, ax4 = plt.subplots(len(fem_g), len(mesh_g), figsize=(len(mesh_g) * 5, len(fem_g) * 5), squeeze=False)
        f3.set_tight_layout(True)
        f4.set_tight_layout(True)

    for i, g in enumerate(mesh_g):
        mesh = g.solve_config(g())
        print("nodes: ", mesh.mesh.coordinates().shape)
        fem_instances = [t(1, g, 3) for t in fem_g]

        for j, e in enumerate(fem_instances):
            for sol in e:
                sol = e.solve_config(sol, debug=True)

                if isinstance(e, ElectricsRandomChargeGenerator):
                    q1 = "u"
                    q2, q3 = "E_x", "E_y"
                elif isinstance(e, MagneticsRandomCurrentGenerator):
                    q1 = "A"
                    q2, q3 = "B_x", "B_y"
                elif isinstance(e, ElasticityFixedLineGenerator):
                    q1 = "m"
                    q2, q3 = "u_x", "u_y"

                if "bdr" in plot:
                    vis = Visualization(
                        sol.quantities["x"], sol.quantities["y"], axis_off=True, visible_area=sol.visible_area
                    )

                    if "mesh" in plot:
                        vis.plot_mesh(sol.mesh.mesh)

                    cmap = ListedColormap(["black", "#009E73"])
                    vis.plot_on_grid(sol.quantities["bdr_v0"], scatter_only=True, cmap=cmap, s=1000.0)

                    ax1[j, i].imshow(vis.to_numpy())
                    ax1[j, i].axis("off")

                if "q" in plot:
                    vis = Visualization(
                        sol.quantities["x"], sol.quantities["y"], axis_off=True, visible_area=sol.visible_area
                    )
                    vis.plot_on_grid(sol.quantities[q1])

                    ax[j, i].imshow(vis.to_numpy())
                    ax[j, i].axis("off")
                if "g_field" in plot:
                    vis = Visualization(
                        sol.quantities["x"], sol.quantities["y"], axis_off=True, visible_area=sol.visible_area
                    )
                    vis.quiver_on_grid(sol.quantities[q2], sol.quantities[q3], interpolate=False)
                    vis.grad_field(sol.quantities[q2], sol.quantities[q3])

                    ax2[j, i].imshow(vis.to_numpy())
                    ax2[j, i].axis("off")
                if "bdr_dist" in plot:
                    sol = map_extend(sol)

                    vis = Visualization(
                        sol.quantities["x"], sol.quantities["y"], axis_off=True, visible_area=sol.visible_area
                    )
                    vis.plot_on_grid(sol.quantities["dist_border"])
                    vis.quiver_on_grid(
                        sol.quantities["dist_border_x"], sol.quantities["dist_border_y"], interpolate=False
                    )

                    vis2 = Visualization(
                        sol.quantities["x"], sol.quantities["y"], axis_off=True, visible_area=sol.visible_area
                    )
                    vis2.plot_on_grid(sol.quantities["dist_bc"])
                    vis2.quiver_on_grid(sol.quantities["dist_bc_x"], sol.quantities["dist_bc_y"], interpolate=False)

                    ax3[j, i].imshow(vis.to_numpy())
                    ax3[j, i].axis("off")

                    ax4[j, i].imshow(vis2.to_numpy())
                    ax4[j, i].axis("off")

                break

    if show:
        plt.show()


def plot_result_compare(artifact_es: str, artifact_ms: str, plot_n: int = 1, fontsize: int = 25) -> None:
    """Compare results saved as wandb artifacts"""
    import matplotlib.pyplot as plt

    table1 = wandb.use_artifact(artifact_es).get("visualizations")
    table2 = wandb.use_artifact(artifact_ms).get("visualizations")

    _, ax = plt.subplots(2 * plot_n, 4, figsize=(12, 6 * plot_n), squeeze=False)

    for i in range(2 * plot_n):
        for j in range(4):
            ax[i][j].axis("off")

    ax[0][0].set_title("El. Potential (pred)", fontsize=fontsize)
    ax[0][1].set_title("El. Potential (gt)", fontsize=fontsize)
    ax[0][2].set_title("El. Field (pred)", fontsize=fontsize)
    ax[0][3].set_title("El. Field (gt)", fontsize=fontsize)

    ax[plot_n][0].set_title("Magn. Potential (pred)", fontsize=fontsize)
    ax[plot_n][1].set_title("Magn. Potential (gt)", fontsize=fontsize)
    ax[plot_n][2].set_title("Magn. Field (pred)", fontsize=fontsize)
    ax[plot_n][3].set_title("Magn. Field (gt)", fontsize=fontsize)

    for i in range(2):
        tbl = table1 if i == 0 else table2

        for next_plot in range(plot_n):
            for plot_image in range(4):
                ax[i * plot_n + next_plot][plot_image].imshow(tbl.data[next_plot][plot_image + 1].image)

    plt.tight_layout()
