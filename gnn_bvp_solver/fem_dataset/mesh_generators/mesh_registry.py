from gnn_bvp_solver.fem_dataset.mesh_generators.cylinder_mesh import CylinderGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.cylinder_precomputed import CylinderMeshPrecomputedGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.disk_mesh import UnitDiskGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.disk_precomputed import UnitDiskPrecomputedGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.l_mesh import LMeshGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.l_mesh_precomputed import LMeshPrecomputedGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import BaseMeshConfig, MeshGeneratorBase
from gnn_bvp_solver.fem_dataset.mesh_generators.square_mesh import UnitSquareGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.u_mesh import UMeshGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.u_mesh_precomputed import UMeshPrecomputedGenerator


def get_mesh(config: BaseMeshConfig) -> MeshGeneratorBase:
    """This function stores a list of mesh generators accessibly via names"""
    if config.mesh_name == "UnitSquareGenerator":
        return UnitSquareGenerator.solve_config(config)
    elif config.mesh_name == "UnitDiskGenerator":
        return UnitDiskGenerator.solve_config(config)
    elif config.mesh_name == "CylinderGenerator":
        return CylinderGenerator.solve_config(config)
    elif config.mesh_name == "LMeshGenerator":
        return LMeshGenerator.solve_config(config)
    elif config.mesh_name == "UMeshGenerator":
        return UMeshGenerator.solve_config(config)
    elif config.mesh_name == "LMeshPrecomputedGenerator":
        return LMeshPrecomputedGenerator.solve_config(config)
    elif config.mesh_name == "UMeshPrecomputedGenerator":
        return UMeshPrecomputedGenerator.solve_config(config)
    elif config.mesh_name == "UnitDiskPrecomputedGenerator":
        return UnitDiskPrecomputedGenerator.solve_config(config)
    elif config.mesh_name == "CylinderMeshPrecomputedGenerator":
        return CylinderMeshPrecomputedGenerator.solve_config(config)
    else:
        raise ValueError(f"Mesh generator {config.mesh_name} not found")
