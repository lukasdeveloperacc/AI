from common.base import BaseTrainer, BaseDataset, BaseExperimentLogger
from common.utils import MetricUtil
from tqdm import tqdm

import torch, logging


class UnetTrainer(BaseTrainer):
    def __init__(
        self,
        dataset: BaseDataset | list[BaseDataset],
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int = 0,
        batch_size: int = 32,
        including_test: bool = True,
        device: str | None = None,
        scheduler=None,
        logger: BaseExperimentLogger = None,
        checkpoint_dir: str = "outputs/checkpoints",
        pretrained_checkpoint_path: str = "",
        is_export_onnx: bool = False,
        is_export_tensorrt: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            network=network,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            batch_size=batch_size,
            including_test=including_test,
            device=device,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            is_export_onnx=is_export_onnx,
            is_export_tensorrt=is_export_tensorrt,
        )

    def train_one_epoch(self, epoch) -> dict:
        try:
            self._network.train()

            log_dict: dict = {}
            image: torch.Tensor
            mask: torch.Tensor
            data_class: torch.Tensor
            loader = self._dataset.load_train_dataset(self._batch_size)
            total_loss: float = 0.0

            for image, mask, data_class in tqdm(loader, desc=f"Trainset Epoch : {epoch}"):
                image = image.to(self._device)
                mask = mask.to(self._device)
                data_class = data_class.to(self._device)
                logging.debug(f"Image shape : {image.shape if isinstance(image, torch.Tensor) else None}")
                logging.debug(f"Mask shape : {mask.shape if isinstance(mask, torch.Tensor) else None}")
                logging.debug(f"Class : {[ self._dataset.classes[i] for i in data_class.view(-1).tolist() ]}")

                if self._example_input is None:
                    self._example_input = image.detach()

                pred: torch.Tensor = self._network(image)
                logging.debug(f"Pred shape : {pred.shape}")

                if mask.ndim == 4:
                    mask = mask.squeeze(1)
                    logging.debug(f"After squeezing, mask shape : {mask.shape}")

                loss: torch.Tensor = self._loss_fn(pred, mask)
                total_loss += loss.item()
                logging.debug(f"Running loss : {total_loss}")

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            log_dict.update({"train_loss": f"{(total_loss / len(loader)):4f}"})

            return log_dict

        except Exception as e:
            logging.error(e)
            return None

    def validate(self, epoch) -> dict:
        try:
            self._network.eval()
            log_dict: dict = {}
            image: torch.Tensor
            mask: torch.Tensor
            data_class: torch.Tensor
            loader = self._dataset.load_valid_dataset(batch_size=self._batch_size)
            total_loss: float = 0.0

            with torch.no_grad():
                for image, mask, data_class in tqdm(loader, desc=f"Validationset Epoch : {epoch}"):
                    image = image.to(self._device)
                    mask = mask.to(self._device)
                    data_class = data_class.to(self._device)
                    logging.debug(f"Image shape : {image.shape if isinstance(image, torch.Tensor) else None}")
                    logging.debug(f"Mask shape : {mask.shape if isinstance(image, torch.Tensor) else None}")
                    logging.debug(f"Class : {[ self._dataset.classes[i] for i in data_class.view(-1).tolist() ]}")

                    if mask.ndim == 4:
                        mask = mask.squeeze(dim=1)

                    pred: torch.Tensor = self._network(image)
                    logging.debug(f"Pred shape : {pred.shape}")
                    loss: torch.Tensor = self._loss_fn(pred, mask)
                    total_loss += loss.item()
                    logging.debug(f"Running loss : {total_loss}")

            log_dict.update({"val_loss": f"{(total_loss / len(loader)):4f}"})

            return log_dict

        except Exception as e:
            logging.error(e)
            return None

    def test(self) -> dict:
        try:
            self._network.eval()
            self.load_checkpoint(self._pretrained_checkpoint_path)

            log_dict: dict = {}
            image: torch.Tensor
            mask: torch.Tensor
            data_class: torch.Tensor
            loader = self._dataset.load_test_dataset(self._batch_size)
            total_loss: float = 0.0
            total_miou: float = 0.0

            with torch.no_grad():
                for image, mask, data_class in tqdm(loader, desc=f"Testset"):
                    image = image.to(self._device)
                    mask = mask.to(self._device)
                    data_class = data_class.to(self._device)
                    logging.debug(f"Image shape : {image.shape if isinstance(image, torch.Tensor) else None}")
                    logging.debug(f"Mask shape : {mask.shape if isinstance(image, torch.Tensor) else None}")
                    logging.debug(f"Class : {[ self._dataset.classes[i] for i in data_class.view(-1).tolist() ]}")

                    if self._example_input is None:
                        self._example_input = image.detach()

                    if mask.ndim == 4:
                        mask = mask.squeeze(1)

                    pred: torch.Tensor = self._network(image)
                    loss: torch.Tensor = self._loss_fn(pred, mask)
                    logging.debug(f"Pred shape : {pred.shape}")
                    pred = torch.argmax(pred, dim=1)
                    logging.debug(f"Pred shape2 : {pred.shape}")
                    miou: torch.Tensor = MetricUtil.compute_miou(pred, mask, len(self._dataset.classes))

                    total_loss += loss.item()
                    total_miou += miou.item()

                    logging.debug(f"Running loss : {total_loss}")
                    logging.debug(f"Running miou : {total_miou}")

            log_dict.update(
                {"test_loss": f"{(total_loss / len(loader)):4f}", "test_miou": f"{(total_miou / len(loader)):4f}"}
            )

            if self._is_export_onnx or self._is_export_tensorrt:
                self.export_onnx()
                if self._is_export_tensorrt:
                    self.export_tensorrt()

            return log_dict

        except Exception as e:
            logging.error(e)
            return None
