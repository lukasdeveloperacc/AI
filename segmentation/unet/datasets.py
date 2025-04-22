from torch.utils.data import Dataset
from torch import Tensor
from typing import List, Optional, Any, Tuple
from PIL import Image
from tqdm import tqdm
from utils import DatasetUtil

import os
import logging
import pandas as pd


class MVTecDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        phase: str,
        data_extensions: List[str],
        domain: str = "segmentation",
        data_frame: Optional[pd.DataFrame] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ):
        self.root_dir = root_dir
        self.phase = phase
        self.data_extensions = data_extensions
        self.domain = domain
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.DataFrame() if data_frame is None else data_frame

        if self.df.empty:
            samples = []
            for root, _, files in tqdm(os.walk(root_dir)):
                for file in files:
                    if isinstance(file, str):
                        name, ext = os.path.splitext(file)
                        ext = ext.replace(".", "")
                        if ext in data_extensions:
                            dataset_phase = os.path.basename(os.path.dirname(root))
                            if dataset_phase != phase:
                                continue

                            label = os.path.basename(root)
                            image_path = os.path.join(root, file)
                            mask_path = image_path.replace(phase, "ground_truth")
                            mask_path = mask_path.replace(file, f"{name}_mask.{ext}")
                            if not os.path.exists(mask_path):
                                mask_path = None
                                logging.debug(f"Mask not found for {image_path}")

                            logging.debug(f"Data path: {image_path}")
                            samples.append({"image_path": image_path, "mask_path": mask_path, "label": label})

            self.df = pd.DataFrame(samples)
            self.train_df, self.test_df = DatasetUtil.train_test_split_from_df(self.df, test_size=0.2, random_state=42)

        logging.info(f"Number of samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, str]:
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        mask = (
            Image.open(row["mask_path"]).convert("L")
            if row["mask_path"] is not None
            else Image.new("L", image.size, color=0)
        )
        label = row["label"]

        logging.debug(f"[Previous transform] Image size: {image.size}, Mask size: {mask.size}, Label: {label}")

        if self.transform:
            image: Tensor = self.transform(image)

        if self.target_transform and self.domain == "classification":
            label = self.target_transform(label)
        elif self.target_transform and self.domain == "segmentation":
            mask: Tensor = self.target_transform(mask)
        else:
            logging.warning(f"Target transform not applied for domain : {self.domain}")

        logging.debug(f"[After transform] Image size: {image.shape}, Mask size: {mask.shape}, Label: {label}")

        return image, mask, label

    def display(self, idx, is_show: bool = True, is_save: bool = False):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        mask = (
            Image.open(row["mask_path"]).convert("L")
            if row["mask_path"] is not None
            else Image.new("L", image.size, color=0)
        )

        if is_show:
            image.show()
            mask.show()
        if is_save:
            output_root = os.path.join(os.getcwd(), "outputs")
            os.makedirs(output_root, exist_ok=True)
            name, ext = os.path.splitext(os.path.basename(row["image_path"]))
            image.save(os.path.join(output_root, f"{row['label']}_{name}{ext}"))
            mask.save(os.path.join(output_root, f"{row["label"]}_maks_{name}{ext}"))

    def get_train_dataset(self):
        return self.__class__(
            root_dir=self.root_dir,
            phase="test",
            data_extensions=self.data_extensions,
            domain=self.domain,
            data_frame=self.train_df,
            transform=self.transform,
            target_transform=self.target_transform,
        )

    def get_test_dataset(self):
        return self.__class__(
            root_dir=self.root_dir,
            phase="test",
            data_extensions=self.data_extensions,
            domain=self.domain,
            data_frame=self.test_df,
            transform=self.transform,
            target_transform=self.target_transform,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset = MVTecDataset(
        root_dir="dataset/mvtec",
        phase="test",
        data_extensions=["jpg", "png"],
        transform=None,
        target_transform=None,
    )
    logging.info(f"Number of test images: {len(dataset)}")

    train_dataset = dataset.get_train_dataset()
    train_dataset.display(0, is_show=False, is_save=True)
    test_dataset = dataset.get_test_dataset()
    test_dataset.display(0, is_show=False, is_save=True)
