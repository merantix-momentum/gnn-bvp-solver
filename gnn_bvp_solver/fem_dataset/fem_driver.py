from typing import Dict
from squirrel.driver import IterDriver
from gnn_bvp_solver.fem_dataset.data_generators.extend_solution import map_extend
from gnn_bvp_solver.fem_dataset.data_generators.generator_base import BaseGeneratorConfig, GeneratorBase
from squirrel.iterstream import Composable, IterableSource
from gnn_bvp_solver.fem_dataset.data_generators.generator_registry import get_gen_and_solve

import numpy as np
import fenics


class FemDriver(IterDriver):
    name = "fem_driver"

    def __init__(
        self,
        train_generator: GeneratorBase,
        val_generator: GeneratorBase = None,
        test_generator: GeneratorBase = None,
        **kwargs
    ) -> None:
        """Initialize the FemDriver.

        Args:
            train_generator (GeneratorBase): Generator for training data.
            val_generator (GeneratorBase): Generator for validation data.
            test_generator (GeneratorBase): Generator for testing data.
            **kwargs: Other keyword arguments passes to super class initializer.
        """
        super().__init__(**kwargs)
        self.generators = {"train": train_generator, "val": val_generator, "test": test_generator}

    @staticmethod
    def map_sample(config: BaseGeneratorConfig) -> Dict:
        """Simplified method that expects all generators to output the same quantities"""
        fenics.set_log_active(False)

        solution = get_gen_and_solve(config)
        solution = map_extend(solution)

        return {
            "data_x": np.stack([solution.quantities[k] for k in solution.io_mapping["x"]], axis=1),
            "data_y": np.stack([solution.quantities[k] for k in solution.io_mapping["y"]], axis=1),
            "edge_index": solution.quantities["edge_index"],
        }

    def get_iter(self, split: str, solve_pde: bool = True, **kwargs) -> Composable:
        """Create iterstream based on dataset split (train, val, test). Applies hooks before loading samples."""
        assert split in ["train", "val", "test"]

        it = IterableSource(iter(self.generators[split]))

        if solve_pde:
            return it.map(self.map_sample)
        else:
            return it
