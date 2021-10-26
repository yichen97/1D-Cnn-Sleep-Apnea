import os
import torch
import pandas as pd
from torch.utils.data import Dataset

# %%
prefix = [0, 44582, 90400, 141442, 186256, 240220, 287142, 338708, 391034, 433376, 482146, 535624, 587030, 629992, 679484, 726312, 774268, 829238, 883526, 933234, 979610, 1034228]


# %%

class MyData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.paths = os.listdir(root_dir)
        self.datasets = []
        for path in self.paths:
            self.datasets.append(pd.read_csv(root_dir + path))

    def __getitem__(self, idx):
        for i in range(len(prefix)):
            if idx < prefix[i]:
                signal = self.datasets[i - 1].iloc[idx - prefix[i - 1], 0:1408]
                label = self.datasets[i - 1].iloc[idx - prefix[i - 1], 1408]
                break
        signal = torch.tensor(signal)
        label = torch.tensor(label)
        return signal.to(torch.float), label.long()

    def __len__(self):
        return prefix[-1]


if __name__ == "__main__":
    dataSet = MyData("D:\\project\\python\\myDesign\\CNN_sleep_apnea_pytorch\\data\\data_smote\\")

