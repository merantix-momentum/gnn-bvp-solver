from gnn_bvp_solver.fem_dataset.data_generators.condenser_plates import CondenserPlatesGenerator
from gnn_bvp_solver.fem_dataset.data_generators.elasticity_fixed_line import ElasticityFixedLineGenerator
from gnn_bvp_solver.fem_dataset.data_generators.electrics_random_charge import ElectricsRandomChargeGenerator
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import BaseGeneratorConfig
from gnn_bvp_solver.fem_dataset.data_generators.magnetics_random_current import MagneticsRandomCurrentGenerator
from gnn_bvp_solver.fem_dataset.data_generators.plates_and_charges import PlatesAndChargesGenerator
from gnn_bvp_solver.fem_dataset.data_generators.rotating_charges import RotatingChargesGenerator
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import GeneratorBase


def get_gen_and_solve(config: BaseGeneratorConfig) -> GeneratorBase:
    """This function stores a list of generators accessibly via names"""
    if config.generator_name == "CondenserPlatesGenerator":
        return CondenserPlatesGenerator.solve_config(config)
    elif config.generator_name == "ElasticityFixedLineGenerator":
        return ElasticityFixedLineGenerator.solve_config(config)
    elif config.generator_name == "ElectricsRandomChargeGenerator":
        return ElectricsRandomChargeGenerator.solve_config(config)
    elif config.generator_name == "MagneticsRandomCurrentGenerator":
        return MagneticsRandomCurrentGenerator.solve_config(config)
    elif config.generator_name == "PlatesAndChargesGenerator":
        return PlatesAndChargesGenerator.solve_config(config)
    elif config.generator_name == "RotatingChargesGenerator":
        return RotatingChargesGenerator.solve_config(config)
    else:
        ValueError(f"generator {config.generator_name} not found")
