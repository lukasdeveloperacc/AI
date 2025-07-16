from common.base import BaseTrainer, BaseDataset
from common.loggers import MLflowLogger
from omegaconf import DictConfig, OmegaConf
from segmentation.unet.config.config import TrainConfig

import common.datasets as dataset_pkg
import segmentation.unet.networks as network_pkg
import segmentation.unet.trainer as trainer_pkg
import logging, torch, hydra


@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    logging.info(f"Before : {cfg}")
    train_config = TrainConfig(**OmegaConf.to_container(cfg, resolve=True))
    logging.info(f"Launching training with config:\n{train_config}")

    datasets = []
    for name in train_config.dataset:
        parameters = train_config.dataset.get(name).__dict__
        dataset: BaseDataset = getattr(dataset_pkg, name)(**parameters)
        datasets.append(dataset)

    if len(datasets) > 1:
        all_classes = []
        for dataset in datasets:
            all_classes += dataset.classes
        all_classes = sorted(set(all_classes))

        for dataset in datasets:
            dataset.classes = all_classes

        dataset = torch.utils.data.ConcatDataset(datasets)
        n_classes = len(all_classes)

    else:
        dataset = datasets[0]
        n_classes = len(dataset.classes)

    logging.info(f"Set dataset : {dataset}")

    name = next(iter(train_config.network))
    parameters = train_config.network.get(name).__dict__
    network: torch.nn.Module = getattr(network_pkg, name)(n_classes=n_classes, **parameters)

    logging.info(f"Set network : {network}")

    name = next(iter(train_config.loss))
    parameters = train_config.loss.get(name)
    if parameters is not None:
        loss_fn: torch.nn.Module = getattr(torch.nn, name)(**parameters.__dict__)
    else:
        loss_fn: torch.nn.Module = getattr(torch.nn, name)()

    logging.info(f"Set loss_fn : {loss_fn}")

    name = next(iter(train_config.optimizer))
    parameters = train_config.optimizer.get(name)
    if parameters is not None:
        optimizer: torch.optim.Optimizer = getattr(torch.optim, name)(network.parameters(), **parameters.__dict__)
    else:
        optimizer: torch.optim.Optimizer = getattr(torch.optim, name)(network.parameters())

    logging.info(f"Set optimizer : {optimizer}")

    scheduler = None

    name = next(iter(train_config.trainer))

    train_config.trainer.get(name).dataset = dataset
    train_config.trainer.get(name).network = network
    train_config.trainer.get(name).loss_fn = loss_fn
    train_config.trainer.get(name).optimizer = optimizer
    train_config.trainer.get(name).scheduler = scheduler

    parameters = train_config.trainer.get(name).__dict__
    logger = MLflowLogger(run_name="second-experiment", tracking_uri="http://172.30.1.149:5000", **parameters)
    train_config.trainer.get(name).logger = logger
    trainer: BaseTrainer = getattr(trainer_pkg, name)(**parameters)

    logging.info(f"Set trainer : {trainer}")

    trainer.train()


if __name__ == "__main__":
    main()
