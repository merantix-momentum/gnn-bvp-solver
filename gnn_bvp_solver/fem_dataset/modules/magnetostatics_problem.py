from typing import Dict, Iterable, Tuple
import fenics as fn
import numpy as np
from gnn_bvp_solver.fem_dataset.modules.fem_problem import FemProblem
from gnn_bvp_solver.fem_dataset.recursive_user_expression import RecursiveUserExpression
from gnn_bvp_solver.fem_dataset.utils import extract_edges_from_triangle_mesh


class MagnetostaticsProblem(FemProblem):
    def __init__(self, mesh: fn.Mesh):
        """Init Magnetostatics Problem. Generate recursive expressions for all quantities to
           enable adding / removing components flexibly. Fem mesh is fixed to unit square for now.

        Args:
            mesh: FEM mesh to evaluate on.
        """
        self.current_density = RecursiveUserExpression(0.0)
        self.permeability = RecursiveUserExpression(1.0)
        super().__init__(mesh)

    def solve(self) -> Dict[str, np.array]:
        """Using fenics to find the solution for an magnetostatics problem

        Returns:
            FEMSolution: solution for magnetic potential and field
        """
        V = fn.FunctionSpace(self.mesh, "P", 2)

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

        bdr_bc = fn.DirichletBC(V, fn.Constant(0.0), TempClass._bdr)
        bcs = [bdr_bc]

        A = fn.TrialFunction(V)
        v = fn.TestFunction(V)
        a = fn.dot(fn.grad(A), fn.grad(v)) * fn.dx
        L = self.permeability * self.current_density * v * fn.dx
        A_res = fn.Function(V)
        fn.solve(a == L, A_res, bcs)

        B_x = fn.project(A_res.dx(1)).compute_vertex_values()
        B_y = fn.project(-A_res.dx(0)).compute_vertex_values()

        return {
            "x": np.array(self.mesh.coordinates()[:, 0], dtype=np.float32),
            "y": np.array(self.mesh.coordinates()[:, 1], dtype=np.float32),
            "I": np.array(fn.project(self.current_density, V).compute_vertex_values(), dtype=np.float32),
            "A": np.array(A_res.compute_vertex_values(), dtype=np.float32),
            "B_x": np.array(B_x, dtype=np.float32),
            "B_y": np.array(B_y, dtype=np.float32),
            "mu": np.array(fn.project(self.permeability, V).compute_vertex_values(), dtype=np.float32),
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
        return {"x": ["x", "y", "I", "bdr_v0", "mu"], "y": ["A", "B_x", "B_y"]}

    @staticmethod
    def physical_quantities() -> Iterable[Tuple]:
        """Return physical quantities present in this simulation.

        Returns:
            Iterable: all quantities (can be 1d or nd)
        """
        return [("I",), ("mu",), ("A",), ("B_x", "B_y")]
