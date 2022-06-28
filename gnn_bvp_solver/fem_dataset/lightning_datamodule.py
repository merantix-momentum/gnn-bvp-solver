from typing import Iterable, Union
from torch.utils.data import IterableDataset
from torch_geometric.loader.dataloader import DataLoader
import pytorch_lightning as pl

from torch_geometric.data import Data
from squirrel.iterstream.torch_composables import TorchIterable


data_type = Union[TorchIterable, Iterable[Data], IterableDataset]


class FEMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data: data_type,
        val_data: data_type = None,
        test_data: data_type = None,
        batch_size_train: int = 1,
        num_workers: int = 0,
    ):
        """Load fem graph data.

        Args:
            train_data (data_type): Training data.
            val_data (data_type, optional): Validation data. Defaults to None.
            test_data (data_type, optional): Test data. Defaults to None.
            batch_size_train (int, optional): Batch size for training. Defaults to 1.
            num_workers (int, optional): Number of workers. Defaults to 0.
        """
        super().__init__()

        self.batch_size = 1  # need bs 1 for val and test to visualize results
        self.batch_size_train = batch_size_train
        self.num_workers = num_workers

        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = test_data

    def _dataloader(self, data: data_type, batch_size: int = None, num_workers: int = None) -> DataLoader:
        """Internal function to create dataloaders.

        Args:
            data (data_type): data source
            batch_size (int): batch size
            num_workers (int): number of workers size

        Returns:
            DataLoader: _description_
        """

        if num_workers is None:
            num_workers = self.num_workers
        if batch_size is None:
            batch_size = self.batch_size

        # shuffle is handled outside
        return DataLoader(data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    def train_dataloader(self) -> DataLoader:
        """Get Train Dataloader.

        Returns:
            DataLoader: Train Dataloader
        """
        return self._dataloader(self.train_dataset, batch_size=self.batch_size_train)

    def val_dataloader(self) -> DataLoader:
        """Get Val Dataloader.

        Returns:
            DataLoader: Val Dataloader
        """
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        """Get Test Dataloader.

        Returns:
            DataLoader: Test Dataloader
        """
        return self._dataloader(self.test_dataset)
