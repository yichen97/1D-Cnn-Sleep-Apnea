# 论文复现

复现了论文 [https://github.com/yichen97/1D-Cnn-Sleep-Apnea](https://arxiv.org/abs/2105.00528v1), 中`1d-cnn`的部分，结果如论文所述。

论文创新点在于指出更短时间输入在一维模型上可以取得更好的效果。

试图将使用小波变换后的频谱图作为输入， 以 `2d-cnn` 的方式训练模型，收敛慢、运行时间长，放弃。

数据可见于论文中，原文数据处理使用Matlab, 数据过大，就不上传了，需要的可以找我要。