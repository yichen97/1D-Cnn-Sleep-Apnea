import os
import torch
import pandas as pd
from torch.utils.data import Dataset

# %%
folder = "D:\\project\\python\\myDesign\\CNN_sleep_apnea_pytorch\\data\\data_smote"
files = os.listdir(folder)

idx_list = []
i = 0
for file in files:
    print("正在载入: " + str(i))
    i = i + 1
    idx_list.append(len(pd.read_csv("data\\data_smote\\" + file)))

prefix = [0]

for i in range(len(idx_list)):
    prefix.append(prefix[i] + idx_list[i])


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
