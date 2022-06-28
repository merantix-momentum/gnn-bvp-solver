from typing import Dict
import torch
from torch_geometric.data import Data
from squirrel.driver.msgpack import MessagepackDriver


class MsgIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, path: str, split: str = None, dry_run: bool = False, shuffle: bool = False) -> None:
        """Creates a PyTorch iterable dataset from a squirrel messagepack driver"""
        if split is None:
            self.driver = MessagepackDriver(path)
        else:
            self.driver = MessagepackDriver(f"{path}/norm_{split}")
        self.dry_run = dry_run
        self.shuffle = shuffle

    def _mapping_f(self, item: Dict) -> Data:
        """Map numpy arrays from squirrel to pt geometric Data"""
        edge_index = torch.tensor(item["edge_index"])
        return Data(x=torch.tensor(item["data_x"]), y=torch.tensor(item["data_y"]), edge_index=edge_index)

    def __iter__(self):
        """Iterate dataset"""
        if self.dry_run:
            it = self.driver.get_iter(max_workers=1, prefetch_buffer=1, shuffle_key_buffer=1, shuffle_item_buffer=1)
        elif self.shuffle:
            it = self.driver.get_iter(max_workers=4, prefetch_buffer=5, shuffle_key_buffer=100, shuffle_item_buffer=100)
        else:
            it = self.driver.get_iter(max_workers=4, prefetch_buffer=5, shuffle_key_buffer=1, shuffle_item_buffer=1)

        for i in it.map(self._mapping_f):
            yield i
