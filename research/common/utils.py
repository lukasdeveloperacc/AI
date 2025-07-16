from typing import Tuple
from dataclasses import make_dataclass, field
from typing import Any, Optional

import torch.nn as nn
import pandas as pd
import logging, torch, yaml, inspect


class DatasetUtil:
    def __init__(self):
        pass

    @staticmethod
    def train_test_split_from_df(
        df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        shuffled_df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        end = int(len(shuffled_df) * (1 - test_size))
        train_df = shuffled_df.iloc[:end]
        test_df = shuffled_df.iloc[end:]

        logging.info(f"Train-test split: {len(train_df)} train samples, {len(test_df)} test samples")

        return train_df, test_df


class MetricUtil:
    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_miou(pred_mask: torch.Tensor, true_mask: torch.Tensor, num_classes: int = 2, eps: float = 1e-6):
        ious = []
        pred_mask = pred_mask.view(-1)
        true_mask = true_mask.view(-1)

        for cls in range(num_classes):
            pred_cls = pred_mask == cls
            true_cls = true_mask == cls
            intersection = (pred_cls & true_cls).float().sum()
            union = (pred_cls | true_cls).float().sum()
            iou = (intersection + eps) / (union + eps)
            ious.append(iou)

        return torch.mean(torch.stack(ious))


class ConfigUtil:
    @staticmethod
    def load_yaml(path: str) -> dict:
        cfg = None

        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)

        except Exception as e:
            logging.error(e)

        return cfg

    @staticmethod
    def make_loss_config_class(loss_cls: type[nn.Module]):
        sig = inspect.signature(loss_cls)
        fields = []

        for name, param in sig.parameters.items():
            annotation = param.annotation if param.annotation != inspect._empty else Any
            default = param.default if param.default != inspect._empty else field(default=None)
            fields.append((name, annotation, default))

        cls_name = f"{loss_cls.__name__}Config"
        return make_dataclass(cls_name, fields)
