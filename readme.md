# Source code for paper "Learning the Solution Operator of Boundary Value Problems using Graph Neural Networks".

## Using the data
The data is stored publicly in google buckets in `requester-pays` mode. To access the data and train your models, you need to [include a billing project](https://cloud.google.com/storage/docs/using-requester-pays#using). The raw data can also be found on [huggingface](https://huggingface.co/datasets/winfried/gnn_bvp_solver/tree/main). You would need to adapt the dataset paths in the config files after downloading and afterwards you would be able to run the experiments without accessing the cloud.

## Use weights & biases:
We use weights & biases for logging and experiment tracking. You can create your free account [here](https://wandb.ai/).
Use `wandb login` to login from your python environment as described [here](https://docs.wandb.ai/quickstart).

## Run training:
Make sure all requirements are installed via `pip install -r requirements.txt`. 
If you have trouble installing PyTorch Geometric on your machine make sure to follow the [official instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 
In the config folder, training scripts are provided for all experiments in the paper.
An example command would be:

```
python -m gnn_bvp_solver.app --config configs/task_shape/es_ma.json --no-gpu 
```

## Test a model:
Look for the model you want to test in the weight & biases [artifact store](https://docs.wandb.ai/guides/artifacts).
The best and latest model will be automatically tagged for each run. Pass `test` as task and the model artifact.
It is important to use the same config like for training.
An example command would be:

```
python -m gnn_bvp_solver.app --task test --artifact model-aer8oj02:v1 --config configs/task_shape/es_ma.json --no-gpu
```

## Paper
Please find the full details for our experiments in the [paper](https://arxiv.org/abs/2206.14092).
We published our work at the [AI for Science workshop at ICML 2022](https://openreview.net/forum?id=4vx9FQA7wiC).

## License
This software is licensed under the [MIT License](LICENSE).

## Cite our work
Please cite the [paper](https://arxiv.org/abs/2206.14092) if you use this code in your own work. 

```
@article{lotzsch2022learning,
  title={Learning the Solution Operator of Boundary Value Problems using Graph Neural Networks},
  author={L{\"o}tzsch, Winfried and Ohler, Simon and Otterbach, Johannes S},
  journal={arXiv preprint arXiv:2206.14092},
  year={2022}
}
```
