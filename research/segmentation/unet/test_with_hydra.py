from common.base import BaseTrainer, BaseDataset
from common.loggers import PythonLogger 
from omegaconf import DictConfig, OmegaConf
from segmentation.unet.config.config import TestConfig

import common.datasets as dataset_pkg
import segmentation.unet.networks as network_pkg
import segmentation.unet.trainer as trainer_pkg
import logging, torch, hydra

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="config", config_name="test", version_base="1.3")
def main(cfg: DictConfig):
    logging.debug(f"Before : {cfg}")
    test_config = TestConfig(**OmegaConf.to_container(cfg, resolve=True))
    logging.info(f"Launching testing with config:\n{test_config}")

    datasets = []
    for name in test_config.dataset:
        parameters = test_config.dataset.get(name).__dict__
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

    name = next(iter(test_config.network))
    parameters = test_config.network.get(name).__dict__
    network: torch.nn.Module = getattr(network_pkg, name)(n_classes=n_classes, **parameters)

    logging.info(f"Set network : {network}")

    name = next(iter(test_config.loss))
    parameters = test_config.loss.get(name)
    if parameters is not None:
        loss_fn: torch.nn.Module = getattr(torch.nn, name)(**parameters.__dict__)
    else:
        loss_fn: torch.nn.Module = getattr(torch.nn, name)()

    logging.info(f"Set loss_fn : {loss_fn}")

    name = next(iter(test_config.trainer))

    test_config.trainer.get(name).dataset = dataset
    test_config.trainer.get(name).network = network
    test_config.trainer.get(name).loss_fn = loss_fn

    parameters = test_config.trainer.get(name).__dict__
    logger = PythonLogger(name="second-experiment", level=logging.INFO)
    test_config.trainer.get(name).logger = logger
    trainer: BaseTrainer = getattr(trainer_pkg, name)(**parameters)

    logging.info(f"Set trainer : {trainer}")

    trainer.test()


if __name__ == "__main__":
    main()
