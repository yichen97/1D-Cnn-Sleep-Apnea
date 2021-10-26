import os
import pandas as pd
import torch
from torch.utils.data import Dataset


# %%

class SmoteData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.paths = os.listdir(root_dir)
        self.dataset = pd.read_csv(root_dir + self.paths[0])

    def __getitem__(self, idx):
        signal = torch.tensor(self.dataset.iloc[idx, 0:1408])
        label = torch.tensor(self.dataset.iloc[idx, 1408])
        return signal.to(torch.float), label.long()

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    dataSet = SmoteData("D:\\project\\python\\myDesign\\CNN_sleep_apnea_pytorch\\data\\data_smote\\")

