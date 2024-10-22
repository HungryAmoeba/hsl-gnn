from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader


class OGBGDataModule(LightningDataModule):
    """`LightningDataModule` for the OGB Graph Property Prediction dataset (ogbg-molhiv).

    The OGB datasets are a collection of benchmark datasets for graph property prediction.
    This module provides data loaders for the ogbg-molhiv dataset.
    """

    def __init__(
        self,
        data_dir: str = "dataset/",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `OGBGDataModule`.

        :param data_dir: The data directory. Defaults to `"dataset/"`.
        :param batch_size: The batch size. Defaults to `32`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self) -> None:
        """Download the OGB dataset if needed."""
        # Download data
        PygGraphPropPredDataset(name="ogbg-molhiv", root=self.hparams.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and split into training, validation, and test sets."""
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=self.hparams.data_dir)
            split_idx = dataset.get_idx_split()

            self.data_train = dataset[split_idx["train"]]
            self.data_val = dataset[split_idx["valid"]]
            self.data_test = dataset[split_idx["test"]]

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state."""
        pass


if __name__ == "__main__":
    _ = OGBGDataModule()
