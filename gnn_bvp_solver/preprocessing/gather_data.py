from gnn_bvp_solver.fem_dataset.data_generators.elasticity_fixed_line import ElasticityFixedLineGenerator
from gnn_bvp_solver.fem_dataset.data_generators.electrics_random_charge import ElectricsRandomChargeGenerator
from gnn_bvp_solver.fem_dataset.data_generators.magnetics_random_current import MagneticsRandomCurrentGenerator
from gnn_bvp_solver.fem_dataset.fem_driver import FemDriver
from gnn_bvp_solver.fem_dataset.fem_driver import GeneratorBase
from gnn_bvp_solver.fem_dataset.mesh_generators.cylinder_precomputed import CylinderMeshPrecomputedGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.disk_precomputed import UnitDiskPrecomputedGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.l_mesh_precomputed import LMeshPrecomputedGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.mesh_base import MeshGeneratorBase
from gnn_bvp_solver.fem_dataset.mesh_generators.square_mesh import UnitSquareGenerator
from gnn_bvp_solver.fem_dataset.mesh_generators.u_mesh_precomputed import UMeshPrecomputedGenerator
from squirrel_datasets_core.spark.setup_spark import get_spark
from squirrel_datasets_core.preprocessing.save_shards import save_composable_to_shards
from pyspark.sql import SparkSession


SAMPLES = 2500
SHARDS = 25

fem_generators = [
    lambda mg: ElectricsRandomChargeGenerator(SAMPLES, mg, 3),
    lambda mg: MagneticsRandomCurrentGenerator(SAMPLES, mg, 3),
    lambda mg: ElasticityFixedLineGenerator(SAMPLES, mg, 3),
]


extra_generators = [
    lambda mg: ElectricsRandomChargeGenerator(SAMPLES, mg, 5, 4),
    lambda mg: MagneticsRandomCurrentGenerator(SAMPLES, mg, 5, 4),
    lambda mg: ElasticityFixedLineGenerator(SAMPLES, mg, 5, 4),
]


def generate_config(session: SparkSession, generator: GeneratorBase, mesh_g: MeshGeneratorBase) -> None:
    """Save shards using squirrel for one combination of fem and mesh generator"""
    key = f"{type(generator).__name__}_{mesh_g}"

    fem_driver = FemDriver(generator)
    iter = fem_driver.get_iter("train", solve_pde=False)

    path = f"gs://squirrel-core-public-data/gnn_bvp_solver/{key}"
    # path = f"local/{key}"

    save_composable_to_shards(
        src_it=iter, num_shards=SHARDS, out_url=path, session=session, hooks=[fem_driver.map_sample]
    )


def generate_spark() -> None:
    """Use spark to generate fem simulations on multiple meshes"""
    session = get_spark("gnn-bvp-preprocessing")

    mesh_generators = {
        "square": UnitSquareGenerator(15, False),
        "disk": UnitDiskPrecomputedGenerator(15, session.sparkContext, False),
        "cylinder": CylinderMeshPrecomputedGenerator(session.sparkContext, False),
        "l_mesh": LMeshPrecomputedGenerator(session.sparkContext, False),
        "u_mesh": UMeshPrecomputedGenerator(session.sparkContext, False),
    }

    mesh_generators_rand = {
        "square_rand": UnitSquareGenerator(15, True),
        "disk_rand": UnitDiskPrecomputedGenerator(15, session.sparkContext, True),
        "cylinder_rand": CylinderMeshPrecomputedGenerator(session.sparkContext, True),
        "l_mesh_rand": LMeshPrecomputedGenerator(session.sparkContext, True),
        "u_mesh_rand": UMeshPrecomputedGenerator(session.sparkContext, True),
    }

    for fem_g in fem_generators:
        for mesh_g in mesh_generators:
            generate_config(session, fem_g(mesh_generators[mesh_g]), mesh_g)
        for mesh_g in mesh_generators_rand:
            generate_config(session, fem_g(mesh_generators_rand[mesh_g]), mesh_g)

    # create extra
    for fem_g in extra_generators:
        for mesh_g in mesh_generators:
            generate_config(session, fem_g(mesh_generators[mesh_g]), mesh_g + "_extra")


if __name__ == "__main__":
    generate_spark()
