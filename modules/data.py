import torch
import feather
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from modules.constants import T, N, FEATURE_LIST


class InfantMotionDataset(Dataset):
    def __init__(self, directory, data, streams, xy_data, predict):
        self.directory = directory
        self.data = data
        self.streams = streams
        self.xy_data = xy_data
        self.samples = []

        files = [Path(directory, f'{segment}.feather') for segment in data.segment]
        
        if predict:
            for fpath in files:
                self.samples.append((fpath, 'original'))
        else:
            df_multi = feather.read_dataframe(files[0])
            augment_names = df_multi.index.get_level_values(0).unique()
            for fpath in files:
                for augment_name in augment_names:
                    self.samples.append((fpath, augment_name))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, augment_name = self.samples[idx]
        df_multi = feather.read_dataframe(fpath).xs(augment_name)
        
        all_streams_data = [df_multi.xs(s).values for s in self.streams]
        data_npy = np.stack(all_streams_data, axis=-1) # (T, N*3, S)

        if self.xy_data:
            # Extract x and y coordinates for each node
            x_coords = data_npy[:, 0::3, :] # (T, N, S)
            y_coords = data_npy[:, 1::3, :] # (T, N, S)
            
            # Combine x and y coordinates
            data_reshaped = np.concatenate([x_coords, y_coords], axis=1) # (T, 2*N, S)
            data_reshaped = data_reshaped.transpose(1, 0, 2).reshape(2*len(self.streams), T, N) # (2*S, T, N)
        else:
            # Reshape 3D: (T, N*3, S) -> (T, N, 3, S) -> (3, T, N, S) -> (3*S, T, N)
            data_reshaped = data_npy.reshape((T, N, 3, len(self.streams))).transpose((2, 0, 1, 3)).reshape((3*len(self.streams), T, N))

        X = torch.tensor(data_reshaped, dtype=torch.float32)
        y = torch.tensor(self.data.corrected_age.iloc[idx], dtype=torch.float32)
        fts = torch.tensor(self.data[FEATURE_LIST].iloc[idx].values, dtype=torch.float32)

        return X, y, fts
 
