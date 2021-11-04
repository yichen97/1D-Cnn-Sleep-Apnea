# -*- coding: utf-8 -*-
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from Model import CNN_apnea

from torch import nn
from torch.utils.data import DataLoader
from dataSet import MyData

dataSet = MyData("D:\\project\\python\\myDesign\\CNN_sleep_apnea_pytorch\\data\\data_smote\\")

# length 长度
train_data_size = int(len(dataSet) * 0.8)
test_data_size = len(dataSet) - train_data_size
train_data, test_data = torch.utils.data.random_split(dataSet, [train_data_size, test_data_size])
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=256)
test_dataloader = DataLoader(test_data, batch_size=256)

# 创建网络模型

CNN_apnea = CNN_apnea()
if torch.cuda.is_available():
    CNN_apnea = CNN_apnea.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器

learning_rate = 1e-3
optimizer = torch.optim.Adam(CNN_apnea.parameters(), lr=learning_rate, weight_decay=1e-5)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 20

# 添加tensorboard
writer = SummaryWriter("logs")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))
    time_start = time.time()
    # 训练步骤开始
    CNN_apnea.train()
    for data in train_dataloader:
        signal, label = data
        signal = torch.unsqueeze(signal, 1)
        if torch.cuda.is_available():
            signal = signal.cuda()
            label = label.cuda()
        outputs = CNN_apnea(signal)
        loss = loss_fn(outputs, label)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    time_end = time.time()
    print('train cost', time_end - time_start)

    # 测试步骤开始
    CNN_apnea.eval()
    total_test_loss = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for data in test_dataloader:
            signal, label = data
            signal = torch.unsqueeze(signal, 1)
            if torch.cuda.is_available():
                signal = signal.cuda()
                label = label.cuda()
            outputs = CNN_apnea(signal)
            loss = loss_fn(outputs, label)
            total_test_loss = total_test_loss + loss.item()
            TP += ((outputs.argmax(1) == 1) & (label == 1)).sum()
            TN += ((outputs.argmax(1) == 0) & (label == 0)).sum()
            FN += ((outputs.argmax(1) == 0) & (label == 1)).sum()
            FP += ((outputs.argmax(1) == 1) & (label == 0)).sum()

    p = TP/(TP + FP)
    r = TP/(TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN)/(TP + TN + FP + FN)
    print("result p = {}  r = {} F1 = {} acc = {}".format(p, r, F1, acc))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", acc, total_test_step)
    total_test_step = total_test_step + 1

    # torch.save(CNN_apnea, "CNN_apnea_{}.pth".format(i))
    # print("模型已保存")

writer.close()
