from typing import Dict, Iterable, Tuple
import fenics as fn
import numpy as np
from gnn_bvp_solver.fem_dataset.modules.fem_problem import FemProblem
from gnn_bvp_solver.fem_dataset.recursive_user_expression import RecursiveUserExpression
from gnn_bvp_solver.fem_dataset.utils import extract_edges_from_triangle_mesh


class ElectrostaticsProblem(FemProblem):
    def __init__(self, mesh: fn.Mesh):
        """Init Electrostatics Problem. Generate recursive expressions for all quantities to
           enable adding / removing components flexibly. Fem mesh is fixed to unit square for now.

        Args:
            mesh: FEM mesh to evaluate on.
        """
        self.charge_density = RecursiveUserExpression(0.0)
        self.permittivity = RecursiveUserExpression(1.0)
        super().__init__(mesh)

    def solve(self) -> Dict[str, np.array]:
        """Using fenics to find the solution for an electrostatics problem

        Returns:
            FEMSolution: solution for potential and electric field
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

        u = fn.TrialFunction(V)
        v = fn.TestFunction(V)
        a = fn.dot(fn.grad(u), fn.grad(v)) * self.permittivity * fn.dx
        L = self.charge_density * v * fn.dx
        u = fn.Function(V)
        fn.solve(a == L, u, bcs)

        electric_field = fn.project(-fn.grad(u))
        e_result = electric_field.compute_vertex_values().reshape((2, -1))

        return {
            "x": np.array(self.mesh.coordinates()[:, 0], dtype=np.float32),
            "y": np.array(self.mesh.coordinates()[:, 1], dtype=np.float32),
            "rho": np.array(fn.project(self.charge_density, V).compute_vertex_values(), dtype=np.float32),
            "u": np.array(u.compute_vertex_values(), dtype=np.float32),
            "E_x": np.array(e_result[0], dtype=np.float32),
            "E_y": np.array(e_result[1], dtype=np.float32),
            "epsilon": np.array(fn.project(self.permittivity, V).compute_vertex_values(), dtype=np.float32),
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
        return {"x": ["x", "y", "rho", "bdr_v0", "epsilon"], "y": ["u", "E_x", "E_y"]}

    @staticmethod
    def physical_quantities() -> Iterable[Tuple]:
        """Return physical quantities present in this simulation.

        Returns:
            Iterable: all quantities (can be 1d or nd)
        """
        return [("rho",), ("epsilon",), ("u",), ("E_x", "E_y")]
