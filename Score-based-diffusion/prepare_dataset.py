import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self, TACs_file, parameters_file, transform=None, target_transform=None):
        self.TACs = pd.read_csv(TACs_file, header=None)
        self.parameters = pd.read_csv(parameters_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        parameters = self.parameters.iloc[idx, ].to_numpy()
        TAC = self.TACs.iloc[idx, ].to_numpy()

        if self.transform:
            TAC = self.transform(TAC)
        if self.target_transform:
            parameters = self.target_transform(parameters)
        return parameters, TAC
