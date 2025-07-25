from typing import Tuple
from dataclasses import make_dataclass, field
from typing import Any, Optional
from numpy.strings import lower
from pydantic import BaseModel

import torch.nn as nn
import pandas as pd
import logging, torch, yaml, inspect, os, inspect

    
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
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def create_from_config_to_yaml(obj: BaseModel) -> None:
        class_name: str = obj.__class__.__name__
        save_dir = os.path.dirname(inspect.getfile(obj.__class__))
        filename: str = class_name.removesuffix("Config")
        file_path = f"{save_dir}/{filename}.yaml"
        
        if os.path.exists(file_path):
            logging.info(f"[{ConfigUtil.__class__.__name__}] {filename} already exists")
            return

        with open(file_path, "w") as f:
            if "compose" in obj.__class__.__name__.lower():
                yaml.dump(obj.model_dump(), f, sort_keys=False)
            else:
                yaml.dump({filename: obj.model_dump()}, f, sort_keys=False)

    @staticmethod
    def create_from_config_to_yaml_when_importing() -> None:
        current_module = inspect.getmodule(inspect.stack()[1][0])
        logging.info(f"current module : {current_module}")
        for name, cls in inspect.getmembers(current_module, inspect.isclass):
            if issubclass(cls, BaseModel) and cls.__module__ == current_module.__name__:
                try:
                    instance = cls()
                    ConfigUtil.create_from_config_to_yaml(instance)
                except Exception as e:
                    logging.error(f"[{name}] {cls.__name__}: {e}")

