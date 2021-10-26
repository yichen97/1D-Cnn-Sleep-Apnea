# %%
import os
import pandas as pd

folder = "D:\\project\\python\\myDesign\\CNN_sleep_apnea_pytorch\\data\\temp"
files = os.listdir(folder)

idx_list = []
i = 0
for file in files:
    print("正在载入: " + str(i))
    i = i + 1
    idx_list.append(len(pd.read_csv("data\\temp\\" + file)))

# 将prefix复制到dataSet文件
prefix = [0]

for i in range(len(idx_list)):
    prefix.append(prefix[i] + idx_list[i])

