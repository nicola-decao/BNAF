# BNAF
Pytorch implementation of Block Neural Autoregressive Flow based on our paper:
> De Cao Nicola, Titov Ivan and Aziz Wilker, [Block Neural Autoregressive Flow](http://arxiv.org/abs/1904.04676) (2019)

## Requirements
* **``python>=3.6``** (it will probably work on older versions but I have not tested on them)
* **``pytorch>=1.0.0``**

Optional for visualization and plotting: ``numpy``, ``matplotlib`` and ``tensorboardX``

## Structure
* [bnaf.py](https://github.com/nicola-decao/BNAF/blob/master/bnaf.py): Implementation of Block Neural Normalzing Flow.
* [toy2d.py](https://github.com/nicola-decao/BNAF/blob/master/toy2d.py): Experiments of 2d toy task (density estimation and energy matching).
* [density_estimation.py](https://github.com/nicola-decao/BNAF/blob/master/density_estimation.py): Experiments on density estimation on real datasets.
* [optim](https://github.com/nicola-decao/BNAF/tree/master/optim): A custom extension of `torch.optim.Adam` and `torch.optim.Adamax` with Polyak averaging. A custom extension of `torch.optim.lr_scheduler.ReduceLROnPlateau` with callbacks.
* [data](https://github.com/nicola-decao/BNAF/tree/master/data): Data classes to handle the real datasets.

## Usage
Below, example commands are given for running experiments.

#### Download datasets
Run the following command to download the datasets:
```
./download_datasets.sh
```

#### Run 2D toy density estimation
This example runs density estimation on the `8 Gaussians` dataset using 1 flow of BNAF with 2 layers and 100 hidden units (`50 * 2` since the data dimensionality is 2).
```
python toy2d.py --dataset 8gaussians \    # which dataset to use
                --experiment density2d \  # which experiment to run
                --flows 1 \               # BNAF flows to concatenate
                --layers 2 \              # layers for each flow of BNAF
                --hidden_dim 50 \         # hidden units per dimension for each hidden layer
                --save                    # save the model after training
                --savefig                 # save the density plot on disk
```

![Imgur](https://i.imgur.com/DWVGsyn.jpg)

#### Run 2D toy energy matching
This example runs energy matching on the `t4` function using 1 flow of BNAF with 2 layers and 100 hidden units (`50 * 2` since the data dimensionality is 2).
```
python toy2d.py --dataset t4 \            # which dataset to use
                --experiment energy2d \   # which experiment to run
                --flows 1 \               # BNAF flows to concatenate
                --layers 2 \              # layers for each flow of BNAF
                --hidden_dim 50 \         # hidden units per dimension for each hidden layer
                --save                    # save the model after training
                --savefig                 # save the density plot on disk
```

![Imgur](https://i.imgur.com/o1QR3XO.jpg)

#### Run real dataset density estimation
This example runs density estimation on the `MINIBOONE` dataset using 5 flows of BNAF with 0 layers.
```
python density_estimation.py --dataset miniboone \  # which dataset to use
                             --flows 5 \            # BNAF flows to concatenate
                             --layers 0 \           # layers for each flow of BNAF
                             --hidden_dim 10 \      # hidden units per dimension for each hidden layer
                             --save                 # save the model after training
```

## Citation
```
De Cao Nicola, Titov Ivan, Aziz Wilker,
Block Neural Autoregressive Flow,
35th Conference on Uncertainty in Artificial Intelligence (UAI19) (2019).
```

BibTeX format:
```
@article{bnaf19,
  title={Block Neural Autoregressive Flow},
  author={De Cao, Nicola and
          Titov, Ivan and
          Aziz, Wilker},
  journal={35th Conference on Uncertainty in Artificial Intelligence (UAI19)},
  year={2019}
}
```

## Feedback
For questions and comments, feel free to contact [Nicola De Cao](mailto:nicola.decao@gmail.com).

## License
MIT
