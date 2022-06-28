from typing import Any, Dict, Iterable, Tuple
import fenics as fn
import numpy as np
from gnn_bvp_solver.fem_dataset.modules.fem_problem import FemProblem
from gnn_bvp_solver.fem_dataset.utils import extract_edges_from_triangle_mesh


class ElasticityProblem(FemProblem):
    def __init__(self, mesh: fn.Mesh):
        """Init Linear Elasticity Problem.

        Args:
            mesh: FEM mesh to evaluate on.
        """
        super().__init__(mesh)

    def solve(self) -> Dict[str, np.array]:
        """Using fenics to find the solution for an elasticity problem

        Returns:
            FEMSolution: solution
        """
        # V0 = fn.FunctionSpace(self.mesh, "DG", 0)
        V = fn.VectorFunctionSpace(self.mesh, "Lagrange", 2)

        class TempClass:
            @staticmethod
            def _bdr(x: Tuple[float, float]) -> bool:
                """Recursively define the boundary condition (for now only enforce V=0)
                   It has to be a static method so we use this hack to make it work.
                Args:
                    x (Tuple[float, float]): point to evaluate

                Returns:
                    bool: True if BC is active otherwise False
                """
                return self.boundary_conditions(x)

        bdr_bc = fn.DirichletBC(V, fn.Constant((0.0, 0.0)), TempClass._bdr)
        bcs = [bdr_bc]

        def _eps(v: Any) -> Any:
            return fn.sym(fn.grad(v))

        def _sigma(u: Any) -> Any:
            return 10e1 * fn.tr(_eps(u)) * fn.Identity(2) + 2 * _eps(u)

        f = fn.Constant((0, -1.0))
        u = fn.TrialFunction(V)
        v = fn.TestFunction(V)
        a = fn.inner(_sigma(u), _eps(v)) * fn.dx
        L = fn.inner(f, v) * fn.dx
        u_res = fn.Function(V)
        fn.solve(a == L, u_res, bcs)

        displacement = u_res.compute_vertex_values().reshape(2, -1)
        # fn.plot(u_res, title='Displacement', mode='displacement')

        s = _sigma(u_res) - (1.0 / 3) * fn.tr(_sigma(u_res)) * fn.Identity(2)  # deviatoric stress
        von_Mises = fn.sqrt(3.0 / 2 * fn.inner(s, s))
        V = fn.FunctionSpace(self.mesh, "P", 1)
        von_Mises = fn.project(von_Mises, V)
        # fn.plot(von_Mises, title='Stress intensity')

        u_magnitude = fn.sqrt(fn.dot(u_res, u_res))
        u_magnitude = fn.project(u_magnitude, V)
        # fn.plot(u_magnitude, 'Displacement magnitude')

        return {
            "x": np.array(self.mesh.coordinates()[:, 0], dtype=np.float32),
            "y": np.array(self.mesh.coordinates()[:, 1], dtype=np.float32),
            "u_x": np.array(displacement[0], dtype=np.float32),
            "u_y": np.array(displacement[1], dtype=np.float32),
            "m": np.array(u_magnitude.compute_vertex_values(), dtype=np.float32),
            "stress": np.array(von_Mises.compute_vertex_values(), dtype=np.float32),
            "bdr_v0": np.array(fn.project(self.boundary_conditions, V).compute_vertex_values(), dtype=np.float32),
            "edge_index": np.array(extract_edges_from_triangle_mesh(self.mesh), dtype=np.int64),
        }

    @staticmethod
    def input_output_mapping() -> Dict[str, Iterable[str]]:
        """Specify which quantities are input or output.

        Returns:
            Dict[str, Iterable[str]]: input / output values of this problem
        """
        # x and y must be first by convention
        return {"x": ["x", "y", "bdr_v0"], "y": ["m", "u_x", "u_y"]}

    @staticmethod
    def physical_quantities() -> Iterable[Tuple]:
        """Return physical quantities present in this simulation.

        Returns:
            Iterable: all quantities (can be 1d or nd)
        """
        return [("m",), ("stress",), ("u_x", "u_y")]
