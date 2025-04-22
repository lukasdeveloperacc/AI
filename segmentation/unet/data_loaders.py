from datasets import MVTecDataset
from torchvision.transforms import ToTensor, Compose, Resize
from torch import Tensor
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    dataset = MVTecDataset(
        root_dir="dataset/mvtec",
        phase="test",
        domain="segmentation",
        data_extensions=["jpg", "png"],
        transform=Compose([Resize(576), ToTensor()]),
        target_transform=Compose([Resize(576), ToTensor()]),
    )

    train_dataset = dataset.get_train_dataset()
    logging.info(f"Number of train images: {len(train_dataset)}")
    train_dataset.display(0, is_show=False, is_save=True)
    test_dataset = dataset.get_test_dataset()
    logging.info(f"Number of test images: {len(test_dataset)}")
    test_dataset.display(0, is_show=False, is_save=True)

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    train_feature: Tensor
    train_mask: Tensor
    label: str
    train_feature, train_mask, label = next(iter(train_dataloader))
    logging.info(f"Train {label} feature shape: {train_feature.shape}")
    logging.info(f"Train {label} mask shape: {train_mask.shape}")
