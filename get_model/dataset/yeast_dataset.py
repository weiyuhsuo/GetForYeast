import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import zarr # Import zarr library
import os
import lightning as L # Import Lightning

class YeastZarrDataset(Dataset):
    """Custom Dataset for reading yeast data from a Zarr archive."""
    def __init__(self, zarr_path: str, split: str = 'train', transform: Optional[callable] = None):
        # Open the zarr archive
        try:
            self.root = zarr.open(zarr_path, mode='r')
            print(f"Successfully opened Zarr archive: {zarr_path}")
        except Exception as e:
            print(f"Error opening Zarr archive {zarr_path}: {e}")
            raise e
            
        # 检查必要的数组是否存在
        required_arrays = ['region_motif', 'condition', 'exp_label']
        for array_name in required_arrays:
            if array_name not in self.root:
                raise ValueError(f"Zarr archive {zarr_path} must contain '{array_name}' array.")

        self.region_motif_data = self.root['region_motif']
        self.condition_data = self.root['condition']
        self.exp_label_data = self.root['exp_label']
        
        # 基本的分割逻辑
        if split == 'train':
            self._indices = np.arange(len(self.region_motif_data))
        elif split == 'val' or split == 'test':
             print(f"Warning: Validation/Test split not fully implemented, using full dataset for '{split}'.")
             self._indices = np.arange(len(self.region_motif_data))
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'.")

        self.transform = transform
        print(f"YeastZarrDataset initialized with {len(self)} samples for split '{split}'.")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        # 获取实际索引
        data_idx = self._indices[idx]
        
        # 从zarr数组读取数据
        region_motif = torch.tensor(self.region_motif_data[data_idx], dtype=torch.float32)
        condition = torch.tensor(self.condition_data[data_idx], dtype=torch.float32)
        exp_label = torch.tensor(self.exp_label_data[data_idx], dtype=torch.float32)
        
        # 应用变换（如果有）
        if self.transform:
            region_motif, condition, exp_label = self.transform(region_motif, condition, exp_label)
            
        # 返回模型期望的格式
        return {
            'region_motif': region_motif,
            'condition': condition,
            'exp_label': exp_label.unsqueeze(-1)
        }

# Modify create_dataloader to use the new YeastZarrDataset
# We might not need this function anymore if we use a LightningDataModule
# def create_dataloader(
#     data_path: str,
#     batch_size: int = 32,
#     num_workers: int = 4,
#     transform: Optional[callable] = None,
#     split: str = 'train' # Add split argument
# ) -> DataLoader:
#     # Assume data_path is now the path to the zarr archive
#     dataset = YeastZarrDataset(zarr_path=data_path, transform=transform, split=split)
#     
#     # The original framework's RegionDataModule might use a custom collate_fn
#     # For now, we use the default, but may need a custom collate_fn later
#     # if batching multiple regions per sample or handling variable lengths.
#     
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=(split == 'train'), # Shuffle only for training
#         num_workers=num_workers,
#         pin_memory=True
#     ) 


class YeastDataModule(L.LightningDataModule):
    """A simple LightningDataModule for YeastZarrDataset."""
    def __init__(self, data_path: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_train = None # Will be set in setup
        self.dataset_val = None   # Will be set in setup
        self._train_data_size = None # Store training data size here
        
    def setup(self, stage: str): # Use stage to differentiate setup for fit, validate, test, predict
        # Create datasets for different stages
        if stage == 'fit':
            # For simplicity, using the same zarr file for train and val
            # A real scenario would use separate train/val data or splits
            self.dataset_train = YeastZarrDataset(zarr_path=self.data_path, split='train')
            self.dataset_val = YeastZarrDataset(zarr_path=self.data_path, split='val') # Use 'val' split
            self._train_data_size = len(self.dataset_train) # Store the size
        elif stage == 'validate':
            self.dataset_val = YeastZarrDataset(zarr_path=self.data_path, split='val')
        # Add logic for test and predict stages if needed

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False, # No shuffle for validation
            num_workers=self.num_workers,
            pin_memory=True
        ) # Add test_dataloader and predict_dataloader if needed 