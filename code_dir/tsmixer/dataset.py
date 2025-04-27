import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        past_covariates: list[str],
        known_covariates: list[str],
        seq_length: int,
        pred_length: int,
        stride: int =1,
    ):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.df = df
        self.stride = stride

        total_length = len(df) - seq_length - pred_length + 1
        self.indices = list(range(0, total_length, stride))

        self.target_col = target_col
        self.past_covariates = past_covariates
        self.known_covariates = known_covariates

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_length + self.pred_length

        segment = self.df.iloc[start_idx:end_idx].copy()
        hist_segment = segment.iloc[: self.seq_length]
        future_segment = segment.iloc[self.seq_length :]

        x_hist_target = hist_segment[self.target_col].values.reshape(-1, 1)
        y_target = future_segment[self.target_col].values.reshape(-1, 1)

        # x_extra_hist
        x_extra_hist = hist_segment[self.past_covariates].values
        x_extra_hist_known = hist_segment[self.known_covariates].values

        # x_extra_future
        x_extra_future_past = future_segment[self.past_covariates].values
        x_extra_future_known = future_segment[self.known_covariates].values

        # x_static
        x_static = np.ones(1)

        x_hist = torch.tensor(x_hist_target, dtype=torch.float32)
        x_extra_hist_combined = np.concatenate(
            [x_extra_hist, x_extra_hist_known], axis=1
        )
        x_extra_hist = torch.tensor(x_extra_hist_combined, dtype=torch.float32)

        # torch.tensor
        x_extra_future_combined = np.concatenate(
            [x_extra_future_past, x_extra_future_known], axis=1
        )
        x_extra_future = torch.tensor(x_extra_future_combined, dtype=torch.float32)
        x_static = torch.tensor(x_static, dtype=torch.float32)
        y = torch.tensor(y_target, dtype=torch.float32)

        return {
            "x_hist": x_hist,
            "x_extra_hist": x_extra_hist,
            "x_extra_future": x_extra_future,
            "x_static": x_static,
            "y": y,
        }
