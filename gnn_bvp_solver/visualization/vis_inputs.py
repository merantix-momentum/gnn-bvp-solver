from typing import List, Union
from matplotlib.colors import ListedColormap
from gnn_bvp_solver.fem_dataset.data_generators.electrics_random_charge import ElectricsRandomChargeGenerator
from gnn_bvp_solver.fem_dataset.data_generators.magnetics_random_current import MagneticsRandomCurrentGenerator
from gnn_bvp_solver.fem_dataset.data_generators.elasticity_fixed_line import ElasticityFixedLineGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.square_mesh import UnitSquareGenerator
from gnn_bvp_solver.visualization.plot_graph import Visualization
from gnn_bvp_solver.fem_dataset.data_generators.extend_solution import map_extend

import matplotlib.pyplot as plt


def plot_in_and_outputs(
    plot: List = None,
    interpolate: bool = False,
    plt_mesh: bool = False,
    plot_el: bool = True,
    savefig: Union[str, None] = None,
) -> None:
    """Plot input and output quantities on a square mesh."""

    if plot is None:
        # plot = ["bdr", "g_field", "q", "bdr_dist", "qin1"]
        plot = ["g_field", "q", "qin1"]

    g = UnitSquareGenerator(15)

    mesh = g.solve_config(g())
    print("nodes: ", mesh.mesh.coordinates().shape)

    es = ElectricsRandomChargeGenerator(1, g, 3)
    em = MagneticsRandomCurrentGenerator(1, g, 3)
    el = ElasticityFixedLineGenerator(1, g, 3)
    fem_g = [es, em]

    if plot_el:
        fem_g = [el]

    skip = 1 if el in fem_g else 0
    _, ax = plt.subplots(1, 3 * len(fem_g) - skip, figsize=(7 * len(fem_g) - skip * 2, 3), squeeze=False)

    i = -1
    for _, e in enumerate(fem_g):
        # i = -1
        jj = 0
        cmap = ListedColormap(["black", "#009E73"])

        for sol in e:
            sol = e.solve_config(sol, debug=True)

            if isinstance(e, ElectricsRandomChargeGenerator):
                q1 = "u"
                tq1 = "(b) El. Potential"
                q2, q3 = "E_x", "E_y"
                tq23 = "(c) El. Field"

                qin1 = "rho"
                tqin1 = "(a) Charge"
            elif isinstance(e, MagneticsRandomCurrentGenerator):
                q1 = "A"
                tq1 = "(e) Magn. Potential"
                q2, q3 = "B_x", "B_y"
                tq23 = "(f) Magn. Field"

                qin1 = "I"
                tqin1 = "(d) Current"
            elif isinstance(e, ElasticityFixedLineGenerator):
                q1 = None
                tq1 = None
                q2, q3 = "u_x", "u_y"
                tq23 = "Displacement Field"

                qin1 = "bdr_v0"
                tqin1 = "Boundary condition"

            if "qin1" in plot:
                i += 1

                if qin1 is not None:
                    vis = Visualization(
                        sol.quantities["x"], sol.quantities["y"], axis_off=True, visible_area=sol.visible_area
                    )

                    # if interpolate:
                    #    vis.plot_on_grid(sol.quantities[qin1])
                    # else:
                    vis.plot_on_grid(sol.quantities[qin1], scatter_only=True, cmap=cmap, s=1000.0)

                    if plt_mesh:
                        vis.plot_mesh(mesh.mesh)

                    ax[jj, i].set_title(tqin1)
                    ax[jj, i].imshow(vis.to_numpy())

                ax[jj, i].axis("off")

            if "bdr" in plot:
                vis = Visualization(
                    sol.quantities["x"], sol.quantities["y"], axis_off=True, visible_area=sol.visible_area
                )
                vis.plot_on_grid(sol.quantities["bdr_v0"])

                i += 1
                ax[jj, i].set_title("Boundary condition")
                ax[jj, i].imshow(vis.to_numpy())
                ax[jj, i].axis("off")

            if "bdr_dist" in plot:
                sol = map_extend(sol)

                vis = Visualization(
                    sol.quantities["x"], sol.quantities["y"], axis_off=True, visible_area=sol.visible_area
                )
                vis.plot_on_grid(sol.quantities["dist_border"])
                vis.quiver_on_grid(sol.quantities["dist_border_x"], sol.quantities["dist_border_y"], interpolate=False)

                # vis2 = Visualization(sol.quantities["x"], sol.quantities["y"], axis_off=True,
                #                      visible_area=sol.visible_area)
                # vis2.plot_on_grid(sol.quantities["dist_bc"])
                # vis2.quiver_on_grid(sol.quantities["dist_bc_x"], sol.quantities["dist_bc_y"], interpolate=False)

                i += 1
                ax[jj, i].set_title("Distance to border")
                ax[jj, i].imshow(vis.to_numpy())
                ax[jj, i].axis("off")

                # i += 1
                # ax[j, i].set_title("Distance to boundary")
                # ax[j, i].imshow(vis2.to_numpy())
                # ax[j, i].axis("off")

            if "q" in plot:
                if q1 is not None:
                    i += 1

                    vis = Visualization(
                        sol.quantities["x"], sol.quantities["y"], axis_off=True, visible_area=sol.visible_area
                    )

                    if interpolate:
                        vis.plot_on_grid(sol.quantities[q1])
                    else:
                        vis.plot_on_grid(sol.quantities[q1], scatter_only=True, s=1000.0)

                        if plt_mesh:
                            vis.plot_mesh(mesh.mesh)

                    ax[jj, i].set_title(tq1)
                    ax[jj, i].imshow(vis.to_numpy())

                ax[jj, i].axis("off")

            if "g_field" in plot:
                vis = Visualization(
                    sol.quantities["x"], sol.quantities["y"], axis_off=True, visible_area=sol.visible_area
                )
                vis.quiver_on_grid(sol.quantities[q2], sol.quantities[q3], interpolate=False)

                if interpolate:
                    vis.grad_field(sol.quantities[q2], sol.quantities[q3])
                else:
                    vis.grad_field(sol.quantities[q2], sol.quantities[q3], scatter_only=True, s=1000.0)
                    if plt_mesh:
                        vis.plot_mesh(mesh.mesh)

                i += 1
                ax[jj, i].set_title(tq23)
                ax[jj, i].imshow(vis.to_numpy())
                ax[jj, i].axis("off")

            break

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
