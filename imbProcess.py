import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# smo = SMOTE(random_state=42)

# 可通过radio参数指定对应类别要生成的数据的数量
smo = SMOTE(sampling_strategy=0.2, random_state=42)
# 生成0和1比例为3比1的数据样本
folder = "D:\\project\\python\\myDesign\\CNN_sleep_apnea_pytorch\\data\\csv_processed"
files = os.listdir(folder)

for file in files:
    data = pd.read_csv("data\\csv_processed\\" + file)
    X = data.iloc[:, 0:1408].values
    y = data.iloc[:, 1408].values
    X, y = smo.fit_resample(X, y)
    data = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    data.to_csv("data\\data_smote\\smote_" + str(file).split('_')[2] + ".csv", index=False)


# for file in files:
#     data = pd.read_csv("data\\csv_processed\\" + file)
#     X = data.iloc[:, 0:1408].values
#     y = data.iloc[:, 1408].values
#     X, y = smo.fit_resample(X, y)
#     data = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
#     data.to_csv("data\\data_smote\\smote_" + str(file).split('_')[2] + ".csv", index=False)
#     pass

