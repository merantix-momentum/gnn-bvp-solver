from typing import Dict
from gnn_bvp_solver.fem_trainer.trainer import FEMTraining
import argparse
import json


def create_training(config: Dict, dry_run: bool, profiling: bool, download_data: bool) -> FEMTraining:
    """Create fem trainer from config and other options"""
    return FEMTraining(
        config["data_train"],
        config["data_val"],
        config["data_test"],
        project="gnn_bvp_solver",
        tags=[config["processor"], *config["tags"]],
        dry_run=dry_run,
        batch_size_train=config["batch_size"],
        profiling=profiling,
        download_data=download_data,
    )


def train(training: FEMTraining, config: Dict, cuda: bool) -> None:
    """Call train on trainer using options from config"""
    training.train(
        config["dim"][0],
        config["dim"][1],
        config["processor"],
        augmentation=config["augmentation"],
        epochs=config["epochs"],
        cuda=cuda,
        remove_pos=config["remove_pos"],
    )


def test(training: FEMTraining, artifact: str, cuda: bool) -> None:
    """Call test on trainer"""
    training.test(artifact, cuda=cuda, project="gnn_bvp_solver")


def test_vis(training: FEMTraining, artifact: str, cuda: bool, failure: bool = False) -> None:
    """Start visualization (either visualize largest loss or first few samples)"""
    training.vis_cases(artifact, cuda=cuda, failure=failure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--dry_run", default="")
    parser.add_argument("--task", default="train")
    parser.add_argument("--artifact", default="")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.add_argument("--profiling", dest="profiling", action="store_true")
    parser.add_argument("--download-data", dest="download_data", action="store_true")
    parser.set_defaults(gpu=True, profiling=False, download_data=False)

    args = parser.parse_args()
    print(args)

    with open(args.config) as f:
        config_dict = json.loads(f.read())

    training = create_training(config_dict, args.dry_run == "dry_run", args.profiling, args.download_data)
    if args.task == "train":
        train(training, config_dict, args.gpu)
    elif args.task == "test":
        test(training, args.artifact, args.gpu)
    elif args.task == "vis":
        test_vis(training, args.artifact, args.gpu)
    elif args.task == "vis_failure":
        test_vis(training, args.artifact, args.gpu, True)
    else:
        print("nothing to do")
