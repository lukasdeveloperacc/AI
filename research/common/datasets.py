from common.base import BaseDataset

from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Lambda

import os
import logging
import pandas as pd
import torch


class MVTecDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        data_extensions: list[str],
        data_frame: pd.DataFrame | None = None,
        transform: torch.nn.Module | None = None,
        target_mask_transform: torch.nn.Module | None = None,
        target_label_transform: torch.nn.Module | None = None,
        resize: tuple[int] = (576, 576),
    ):
        self._resize = resize

        super().__init__(
            root_dir, data_extensions, data_frame, transform, target_mask_transform, target_label_transform
        )

    # TODO: Make downlaod function
    def download() -> None:
        return None

    def make_data_frame(self) -> pd.DataFrame:
        samples = []
        mask_dir_name = "ground_truth"
        for root, _, files in tqdm(os.walk(self._root_dir)):
            for file in files:
                if isinstance(file, str):
                    name, ext = os.path.splitext(file)
                    ext = ext.replace(".", "")
                    if ext in self._data_extensions:
                        dataset_phase = os.path.basename(os.path.dirname(root))
                        if dataset_phase == mask_dir_name:
                            continue

                        label = os.path.basename(root)
                        image_path = os.path.join(root, file)
                        mask_path = image_path.replace(dataset_phase, mask_dir_name)
                        mask_path = mask_path.replace(file, f"{name}_mask.{ext}")
                        if not os.path.exists(mask_path):
                            logging.debug(f"Mask not found for {mask_path}")
                            mask_path = None

                        sample = {"image_path": image_path, "mask_path": mask_path, "class": label}
                        logging.debug(f"Sample : {sample}")
                        samples.append(sample)

        return pd.DataFrame(samples)

    def make_transform(self) -> torch.nn.Module:
        return Compose([Resize(self._resize), ToTensor()])

    def make_transform_mask(self) -> torch.nn.Module:
        return Compose([Resize(self._resize), ToTensor()])

    def make_transform_data_class(self, one_hot_encoding: bool = False) -> Lambda:
        if one_hot_encoding:
            import torch.nn.functional as F

            return Lambda(
                lambda data_class: F.one_hot(
                    torch.tensor(self.classes.index(data_class), dtype=torch.long),
                    num_classes=len(self.classes),
                ).float()
            )
        else:
            return Lambda(lambda data_class: torch.tensor(self._classes.index(data_class), dtype=torch.long))

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self._df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")

        mask = (
            Image.open(row["mask_path"]).convert("L")
            if row["mask_path"] is not None
            else Image.new("L", image.size, color=0)
        )
        data_class = row["class"]

        if self._transform:
            image: torch.Tensor = self._transform(image)

        if self._target_data_class_transform:
            data_class: torch.Tensor = self._target_data_class_transform(data_class)

        if self._target_mask_transform:
            mask: torch.Tensor = self._target_mask_transform(mask)

        # Set mask pixels > 0 to data_class, else 0
        mask = (mask > 0).long() * data_class
        logging.debug(
            f"{self.classes[data_class] if hasattr(self, 'classes') and self.classes else data_class} : {set(mask[0].squeeze().view(-1).cpu().numpy().tolist())}"
        )

        return image, mask, data_class


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dataset = MVTecDataset(root_dir="datasets/mvtec", data_extensions=["jpg", "png"], resize=576)

    train_dataset = dataset.load_train_dataset()
    d = next(iter(train_dataset))
