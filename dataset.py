from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
WINDOW_SIZE = 300
from math import exp

class WifiDataset(Dataset):
    def __init__(self, data_dir, window_size=WINDOW_SIZE, stride=WINDOW_SIZE, default_val=90):
        super(WifiDataset, self).__init__()
        self.default_val = default_val
        self.data_dir = data_dir

        posi_cols = ['x', 'y', 'z']
        acce_cols = ['ax', 'ay', 'az']
        gyro_cols = ['gx', 'gy', 'gz']
        magn_cols = ['mx', 'my', 'mz']

        self.data = pd.read_csv(data_dir)
        # if there is a null values left
        self.data.fillna(-default_val, inplace=True)

        wifi_signal_cols = [col for col in self.data.columns if col not in ['x', 'y', 'z', "num_visible_APs", "timestamp",
                                                                     'gx', 'gy', 'gz', 'ax', 'ay', 'az', 'mx', 'my',
                                                                     'mz', '' ,'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1']]

        self.window_size = window_size
        self.stride = stride
        self.wifi_vals = self.data[wifi_signal_cols].values
        self.wifi = self.normalize_wifi(self.wifi_vals)

        self.posi = self.data[posi_cols].values
        self.gyro_vals = self.data[gyro_cols].values
        self.magn_vals = self.data[magn_cols].values
        self.acce_vals = self.data[acce_cols].values

    def __len__(self):
        return (len(self.data) - self.window_size) // self.stride

    def __getitem__(self, idx):
        wifi = self.wifi[(idx * self.stride):(idx * self.stride) + self.window_size]
        acce = self.acce_vals[(idx * self.stride):(idx * self.stride) + self.window_size]
        gyro = self.gyro_vals[(idx * self.stride):(idx * self.stride) + self.window_size]
        magn = self.magn_vals[(idx * self.stride):(idx * self.stride) + self.window_size]
        posi = self.posi[(idx * self.stride):(idx * self.stride) + self.window_size]
        dtype = torch.float
        wifi = np.concatenate((wifi, gyro, acce, magn), axis=1)
        wifi, posi = torch.tensor(wifi, dtype=dtype), torch.tensor(posi, dtype=dtype)
        f_name = self.data_dir.split('/')[-1]
        return wifi, posi, f_name

    def normalize_wifi(self, x):
        x += self.default_val
        x /= self.default_val
        x = np.power(x, exp(1))
        return x

    @staticmethod
    def normalize_imu(x):
        min_val = min(min(x[:, 0]), min(x[:, 1]), min(x[:, 2]))
        max_val = max(max(x[:, 0]), max(x[:, 1]), max(x[:, 2]))
        x -= min_val
        x /= (max_val - min_val)
        return x




