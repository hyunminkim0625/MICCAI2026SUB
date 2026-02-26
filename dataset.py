import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import lightning as L
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from glob import glob
import torch
from typing import Any, Dict
from tqdm import tqdm

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)

def make_train_transform(img_size: int):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size),interpolation=cv2.INTER_LINEAR,
                            scale=(0.8, 1.0), ratio=(0.75, 1.3333), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=0.5),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])

def make_val_test_transform(img_size: int):
    return A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])

class BenchmarkDataset(Dataset):
    def __init__(self, csv_file, transform=None, train=False, hard_label=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train
        self.hard_label = hard_label

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filepath = row['file_path']
        cfp_image = np.array(Image.open(filepath).convert("RGB"))
        cfp_image = self.transform(image=cfp_image)['image']
        img_name = row['Filename']
        label = torch.tensor([row['Avg_Label']]).float()
        if self.hard_label:
            label = torch.tensor([1.0]) if row['Avg_Label'] >= 0.5 else torch.tensor([0.0])
        return {
            "cfp_image": cfp_image,
            "img_name": img_name,
            "label": label,
        }
    
class DataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        img_size: int = 256,
        num_workers: int = 4,
        hard_label : bool = False,
    ):
        super().__init__()
        self.batch_size = int(batch_size)
        self.img_size = int(img_size)
        self.eval_img_size = int(img_size)
        self.num_workers = int(num_workers)
        self._build_transforms()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.hard_label = hard_label


    # ------------------------- helpers ------------------------- #
    def _build_transforms(self):
        self.train_transform = make_train_transform(self.img_size)
        self.val_test_transform = make_val_test_transform(self.eval_img_size)

    # -------------------- Lightning hooks ---------------------- #
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = BenchmarkDataset(
                "filepath'
                transform=self.train_transform,
                train=True,
                hard_label = self.hard_label,
            )
            self.val_dataset = BenchmarkDataset(
                "filepath'
                transform=self.val_test_transform,
            )
        if stage in (None, "test"):
            self.test_dataset = BenchmarkDataset(
                "filepath'
                transform=self.val_test_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
  