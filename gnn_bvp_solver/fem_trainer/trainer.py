from typing import List
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datetime import datetime
from gnn_bvp_solver.fem_dataset.lightning_datamodule import FEMDataModule
from gnn_bvp_solver.fem_dataset.msg_dataset import MsgIterableDataset
from gnn_bvp_solver.fem_trainer.graph_training_module import GNNModule
from squirrel.driver.msgpack import MessagepackDriver, MessagepackSerializer
from squirrel.store import SquirrelStore
from pathlib import Path
from queue import PriorityQueue

import wandb
import pytorch_lightning as pl


class FEMTraining:
    def __init__(
        self,
        data_dir_train: str,
        data_dir_val: str,
        data_dir_test: str,
        project: str = None,
        batch_size_train: int = 1,
        tags: List = None,
        init_wandb: bool = True,
        dry_run: bool = False,
        profiling: bool = False,
        download_data: bool = True,
        seed: int = 111,
    ):
        """Init Fem training. Load data from default directory"""
        self.profiling = profiling
        self.seed = seed
        pl.seed_everything(self.seed)

        if tags is None:
            tags = []

        self.dry_run = dry_run
        if self.dry_run:
            batch_size_train = 1
            tags += ["dry_run"]

        if download_data:
            data_dir_train = self.convert_to_local_path(data_dir_train)
            data_dir_val = self.convert_to_local_path(data_dir_val)
            data_dir_test = self.convert_to_local_path(data_dir_test)
        self.data_module = self.load_msgpack(data_dir_train, data_dir_val, data_dir_test, batch_size_train, dry_run)

        if init_wandb and not self.profiling:
            parent_dir = Path(__file__).parent.parent.parent.resolve()
            load_dotenv(f"{parent_dir}/wandb.env", override=True)

            if tags is None:
                tags = []

            if project is not None:
                wandb.init(tags=tags, project=project)
            else:
                wandb.init(tags=tags)

            wandb.config.seed = self.seed

    def convert_to_local_path(self, path: str) -> str:
        """Cache data locally if we train for multiple epochs."""
        N_SHARD = 100
        local_path = path.replace("gs://", "")
        local_path = "data/" + local_path.replace("/", "-")

        if Path(local_path).exists():
            print(f"directory {local_path} already exists")
            return local_path

        print("downloading data")
        store = SquirrelStore(local_path, serializer=MessagepackSerializer())
        driver = MessagepackDriver(path)
        driver.get_iter().batched(N_SHARD, drop_last_if_not_full=False).map(
            lambda it: store.set(value=list(it))
        ).tqdm().join()

        return local_path

    def load_msgpack(
        self, path_train: str, path_val: str, path_test: str, batch_size_train: int, dry_run: bool
    ) -> FEMDataModule:
        """Create pt lightning datamodule for messagepack dataset."""
        return FEMDataModule(
            train_data=MsgIterableDataset(path_train, dry_run=dry_run, shuffle=True),
            val_data=MsgIterableDataset(path_val, dry_run=dry_run, shuffle=False),
            test_data=MsgIterableDataset(path_test, dry_run=dry_run, shuffle=False),
            batch_size_train=batch_size_train,
        )

    def train(
        self,
        dim_in: int,
        dim_out: int,
        processor: str,
        augmentation: List,
        remove_pos: True,
        epochs: int = 1000,
        cuda: bool = True,
    ) -> None:
        """Train for fixed number of epochs. Weights and logs are saved to wandb."""
        gpus = 1 if cuda else 0
        lightning_module = GNNModule(dim_in, dim_out, processor, augmentation=augmentation, remove_pos=remove_pos)

        if not self.profiling:
            wandb_logger = WandbLogger(
                log_model=True, name="gnn_bvp_logs", version=datetime.today().strftime("%Y-%m-%d")
            )
            wandb_logger.watch(lightning_module.model, log="all")

        if self.dry_run:
            checkpoint_callback = ModelCheckpoint(every_n_epochs=0)
            trainer = pl.Trainer(
                max_epochs=epochs,
                gpus=gpus,
                logger=wandb_logger,
                callbacks=[checkpoint_callback],
                log_every_n_steps=1,
                overfit_batches=1,
            )
        elif self.profiling:
            checkpoint_callback = ModelCheckpoint(every_n_epochs=0)
            trainer = pl.Trainer(
                max_epochs=epochs, gpus=gpus, callbacks=[checkpoint_callback], profiler="simple", limit_val_batches=0
            )
        else:
            checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=3, save_last=True)
            trainer = pl.Trainer(
                max_epochs=epochs, gpus=gpus, logger=wandb_logger, callbacks=[checkpoint_callback], log_every_n_steps=50
            )

        trainer.fit(lightning_module, datamodule=self.data_module)

        if not self.profiling:
            wandb.finish()

    def test(self, artifact_reference: str, cuda: bool = True) -> None:
        """Download model artifact from wandb and test, which requires still to copy the artifact name manually.

        Args:
            artifact_reference (str): Artifact name from wandb console.
            cuda (bool, optional): Use GPU for testing. Defaults to True.
        """
        gpus = 1 if cuda else 0

        wandb.init(project="fem-end-to-end")
        wandb_logger = WandbLogger(name="gnn_bvp_logs", version=datetime.today().strftime("%Y-%m-%d"))

        artifact = wandb.use_artifact(artifact_reference, type="model")
        artifact_dir = artifact.download()
        lightning_module = GNNModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")

        trainer = pl.Trainer(gpus=gpus, logger=wandb_logger, log_every_n_steps=1)
        trainer.test(lightning_module, datamodule=self.data_module)

        wandb.finish()

    def vis_cases(self, artifact_reference: str, cuda: bool = True, failure: bool = False) -> None:
        """Visualize example cases (either random or highest loss of model)"""
        artifact = wandb.use_artifact(artifact_reference, type="model")
        artifact_dir = artifact.download()
        lightning_module = GNNModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")

        if cuda:
            lightning_module = lightning_module.cuda()

        q = PriorityQueue()
        idx = 0
        n = 10

        print("testing ..")
        for batch in self.data_module.test_dataloader():
            if cuda:
                batch = batch.cuda()

            y_hat = lightning_module(batch.x, batch.edge_index)

            # loss = lightning_module.loss_f(y_hat.squeeze(), batch.y)
            # print(y_hat.min(), y_hat.mean(), y_hat.max())
            # print(batch.y.min(), batch.y.mean(), batch.y.max())
            # print(loss)

            # compute loss for the VECTOR FIELDS
            loss = lightning_module.loss_f(y_hat.squeeze()[:, 1:], batch.y[:, 1:])

            loss_it = loss.cpu().item()
            q.put((-loss_it, idx, (batch, y_hat.cpu().detach())))
            idx += 1

            if idx % 5 == 0:
                print(f"\r{idx}", end="")

            if idx >= n and not failure:
                print(f"not in failure mode, break after {n} samples")
                break

        print(f"select {n} examples to visualize")

        columns = ["Loss", "Pred_pot", "GT_pot", "Pred_vec", "GT_vec"]
        table = wandb.Table(columns=columns)

        for _i in range(n):
            item = q.get()
            print("loss", -item[0])

            batch, data_y_hat = item[2]
            data_x = batch.x.cpu()
            data_y = batch.y.cpu()

            potential_predicted, potential_gt, vector_f_predicted, vector_f_gt = lightning_module.vis_out_gt(
                data_x, data_y, data_y_hat, log=False, size_x=10, size_y=8, res=100
            )

            table.add_data(
                -item[0],
                wandb.Image(potential_predicted),
                wandb.Image(potential_gt),
                wandb.Image(vector_f_predicted),
                wandb.Image(vector_f_gt),
            )

        wandb.log({"visualizations": table})
