# %%
import os
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat

root_dir = "D:\\project\\python\\myDesign\\CNN_sleep_apnea_pytorch\\data\\data_label\\"
files = os.listdir(root_dir)
train_files = []
label_files = []
for path in files:
    if re.search("label", path):
        label_files.append(path)
    else:
        train_files.append(path)

print("文件读取完毕")
signal = []
label = []

# %%
# file_name = "ucddb002.mat"
# file_path = root_dir + file_name
# temp = loadmat(file_path)
# signal.append(np.array(temp["signal"]))
for file_name in train_files:
    file_path = root_dir + file_name
    temp = loadmat(file_path)
    signal.append(np.array(temp["signal"]).astype('float32'))
print("Data loaded")

# %%
for file_name in label_files:
    file_path = root_dir + file_name
    temp = loadmat(file_path)
    label.append(np.array(temp["labels"]))
print("Label loaded")

# %%
# 初始化data矩阵
n = 30000

# for i in range(len(signal)):
for i in range(len(signal)):
    idx = 0
    k = 128
    signal_feature = np.zeros((n, 1409), 'float32')
    while k + 1280 <= len(signal[i]):  # 如果还存在11s的信息可以读取，即128 + 1280 = 1408 点信号
        signal_feature[idx, 0:1408] = (signal[i][k - 128:k + 1280]).T  # 前1408是feature, np的索引前闭后开！
        signal_feature[idx, 1408] = (label[i][int(k / 128)])  # 最后一位是label
        idx = idx + 1
        k = k + 128
        # 如果达到了n条数据，或者该路信号已经不足产生n条，则写入文件
        if idx == n - 1 or k + 1280 > len(signal[i]):
            idx = 0
            name = "data\\csv_processed\\data_for_" + str(i) + "th_signal_" + str(k) + "batch.csv"
            data = pd.DataFrame(signal_feature)
            data = data[data.apply(np.sum, axis=1) != 0]
            data.to_csv(name, index=False)
            signal_feature = np.zeros((n, 1409), 'float32')



