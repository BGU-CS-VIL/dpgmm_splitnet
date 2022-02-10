# from typing import Any, Callable, Optional, Tuple

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class SplitDataset(Dataset):
    def __init__(self, X, y, permute: bool = False):
        super().__init__()
        self.X = X
        self.y = y
        self.permute = permute

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x_ = self.X[index]
        # normalize:
        x = torch.Tensor(StandardScaler().fit_transform(x_))
        y = torch.Tensor(self.y[index])

        # points order permutation "augmentation":
        if self.permute:
            permute_idx = torch.randperm(x.shape[0])
            x = x[permute_idx]
            x_ = x_[permute_idx]
            y = y[permute_idx]

        return x, x_, y
