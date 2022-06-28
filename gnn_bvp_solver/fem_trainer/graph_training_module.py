from typing import List, Tuple, Union
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import torchmetrics

from torch_geometric.data.batch import Batch
from gnn_bvp_solver.fem_trainer.models.main_model import MainModel
from gnn_bvp_solver.visualization.plot_graph import Visualization


class GNNModule(pl.LightningModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        processor: str = "unet3",
        processor_hidden: int = 128,
        mlp_hidden: int = 128,
        augmentation: List = None,
        remove_pos: bool = True,
    ):
        """PT Lightning module for simple graph net training with MSE loss & Adam."""
        super().__init__()

        self.save_hyperparameters()
        self.loss_f = torch.nn.MSELoss()
        self.model = MainModel(dim_in, mlp_hidden, processor_hidden, dim_out, processor, augmentation, remove_pos)

        self.mape = torchmetrics.MeanAbsolutePercentageError()

    def relative_absolute_error(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate relative absolute error"""
        return torch.sum(torch.abs(input - target)) / torch.sum(torch.abs(target))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Execute model

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: model output
        """
        return self.model(x, edge_index)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Train model

        Args:
            batch (Batch): batch containing graph structure and data
            batch_idx (int): batch index, currently unused

        Returns:
            torch.Tensor: aggregated loss
        """
        y_hat = self.forward(batch.x, batch.edge_index)
        loss = self.loss_f(y_hat.squeeze(), batch.y)

        self.log("train_loss", loss)
        return loss

    def vis_potential(
        self,
        data_x: np.array,
        data_y: np.array,
        size_x: int = 5,
        size_y: int = 4,
        vis_a: np.array = None,
        res: int = 50,
        vmin: float = None,
        vmax: float = None,
    ) -> None:
        """Visualize potential assuming fixed order of data in tensor"""
        return (
            Visualization(data_x[:, 0], data_x[:, 1], figsize=(size_x, size_y), visible_area=vis_a, resolution=res)
            .plot_on_grid(data_y[:, 0], scatter=False, vmin=vmin, vmax=vmax)
            .to_numpy()
        )

    def vis_vector_field(
        self,
        data_x: np.array,
        data_y: np.array,
        size_x: int = 5,
        size_y: int = 4,
        vis_a: np.array = None,
        res: int = 50,
        vmin: float = None,
        vmax: float = None,
    ) -> None:
        """Visualize vector field assuming fixed order of data in tensor"""
        return (
            Visualization(data_x[:, 0], data_x[:, 1], figsize=(size_x, size_y), visible_area=vis_a, resolution=res)
            .grad_field(data_y[:, 1], data_y[:, 2], vmin=vmin, vmax=vmax)
            .quiver_on_grid(data_y[:, 1], data_y[:, 2], normalize=True, interpolate=False)
            .to_numpy()
        )

    def vis_out_gt(
        self,
        data_x: np.array,
        data_y: np.array,
        data_y_hat: np.array,
        size_x: int = 5,
        size_y: int = 4,
        log: bool = True,
        res: int = 50,
    ) -> Union[Tuple, None]:
        """Visualize both ground truth and predictions"""
        vis_a = None  # optionally infer the visible area here from the tensor

        pot_max = max(data_y[:, 0].max(), data_y_hat[:, 0].max())
        pot_min = min(data_y[:, 0].min(), data_y_hat[:, 0].min())

        ming, maxg = Visualization.get_min_max_grad_field(data_y[:, 1], data_y[:, 2])
        ming_h, maxg_h = Visualization.get_min_max_grad_field(data_y_hat[:, 1], data_y_hat[:, 2])
        v_max = max(maxg, maxg_h)
        v_min = min(ming, ming_h)

        potential_gt = self.vis_potential(
            data_x, data_y, size_x=size_x, size_y=size_y, vis_a=vis_a, res=res, vmin=pot_min, vmax=pot_max
        )
        potential_predicted = self.vis_potential(
            data_x, data_y_hat, size_x=size_x, size_y=size_y, vis_a=vis_a, res=res, vmin=pot_min, vmax=pot_max
        )

        vector_f_gt = self.vis_vector_field(
            data_x, data_y, size_x=size_x, size_y=size_y, vis_a=vis_a, res=res, vmin=v_min, vmax=v_max
        )
        vector_f_predicted = self.vis_vector_field(
            data_x, data_y_hat, size_x=size_x, size_y=size_y, vis_a=vis_a, res=res, vmin=v_min, vmax=v_max
        )

        if log:
            image_potential = np.concatenate([potential_predicted, potential_gt], axis=1)
            image_vector = np.concatenate([vector_f_predicted, vector_f_gt], axis=1)
            data = np.concatenate([image_potential, image_vector], axis=0)  # image_all

            wandb.log({"Field plots (Pred | GT)": wandb.Image(data)})
        else:
            return potential_predicted, potential_gt, vector_f_predicted, vector_f_gt

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
        metric: str = "val_loss",
    ) -> None:
        """Validate model"""
        y_hat = self.forward(batch.x, batch.edge_index)
        loss = self.loss_f(y_hat.squeeze(), batch.y)
        self.log(metric, loss)

        self.log("mse_potential", self.loss_f(y_hat.squeeze()[:, 0], batch.y[:, 0]))
        self.log("mse_vector_field", self.loss_f(y_hat.squeeze()[:, 1:], batch.y[:, 1:]))

        self.mape(y_hat.squeeze(), batch.y)
        self.log("relative_absolute_error", self.relative_absolute_error(y_hat.squeeze(), batch.y))
        self.log("mape", self.mape, on_step=True, on_epoch=True)

        # do not log all validation samples
        # log only every 10 epochs
        if (np.random.randint(200) > 198 or batch_idx == 0) and self.current_epoch % 10 == 0:
            data_x = batch.x.cpu().numpy()
            data_y = batch.y.cpu().numpy()
            data_y_hat = y_hat.detach().cpu().numpy()

            self.vis_out_gt(data_x, data_y, data_y_hat)
            # wandb.log({"Field plots": wandb.Image(img)})

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        """Test model

        Args:
            batch (Batch): batch containing graph structure and data
            batch_idx (int): batch index, currently unused
        """
        self.validation_step(batch, batch_idx, metric="test_loss")
        # set index = 0 to plot all results -> takes long time

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer

        Returns:
            torch.optim.Optimizer: used torch optimizer
        """
        return torch.optim.Adam(self.parameters())
