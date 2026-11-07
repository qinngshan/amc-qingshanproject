# 通信信号调制识别 + 时序异常检测

通信工程 × 机器学习的两个可复现实验项目。全部代码可直接运行，结果来自实际执行。

## 环境

```bash
pip install numpy scikit-learn matplotlib
# 中文 pip 源（国内网络）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scikit-learn matplotlib
```

Python 3.9+ 即可，无需 GPU。

---

## 项目一：调制方式自动识别（AMC）

盲接收机不知道对方使用的是哪种调制方式，这是通信对抗与频谱监测中的实际问题。
本项目从零搭建信号仿真链路，提取特征后用机器学习做分类。

### 运行

```bash
python amc.py            # 完整实验（约 60 秒）
python amc.py --quick    # 减少样本量快速验证
python augment.py        # 数据增强 / 域适应对比
python plot.py           # 生成全部图表到 figures/
```

### 技术方案

| 环节 | 做法 |
|---|---|
| 信号生成 | BPSK / QPSK / 16QAM，根升余弦脉冲成型（β=0.35，4 倍上采样），按符号功率叠加 AWGN |
| 信道损伤 | 载波频偏 ±0.002、相位噪声 0.005~0.03 rad、定时误差 ±0.3 采样点 |
| 特征提取 | 8 维：四阶/六阶累积量 C40、C42、C63、\|C41\|；幅度标准差、峰均比、相位标准差、频谱峭度 |
| 归一化 | 单位功率归一化，避免模型学到的是信噪比而不是调制方式 |
| 分类器 | KNN / SVM(RBF) / 随机森林 |

### 实测结果

**理想信道**

| 模型 | 总体准确率 | SNR ≥ 8dB |
|---|---|---|
| KNN | 98.35% | 100% |
| SVM | 97.61% | 100% |
| 随机森林 | 98.11% | 100% |

**跨信道泛化**（训练用理想信道，测试用含损伤信道）

| 模型 | 准确率 |
|---|---|
| KNN | 63.46% |
| SVM | 54.35% |
| 随机森林 | 70.94% |

> 这是项目里最有价值的一次发现：理想信道下接近满分的模型，
> 一遇到频偏和相位噪声就掉到 63%。说明纯累积量特征对相位损伤非常敏感。

**数据增强后**（训练集混入带损伤样本）

| 训练策略 | 理想信道 | 含损伤信道 |
|---|---|---|
| 仅理想信道训练 | 98.48% | 69.90% |
| 仅含损伤训练 | 96.77% | 94.02% |
| 理想 + 损伤混合 | 97.44% | **94.78%** |

在几乎不损失理想信道性能的前提下，跨信道准确率从 69.9% 提升到 94.8%。

### 图表

- `figures/fig1_snr_accuracy.png` — SNR-准确率曲线 + 域适应前后对比
- `figures/fig2_confusion.png` — 混淆矩阵
- `figures/fig3_importance.png` — 特征重要性排序
- `figures/fig4_constellation.png` — 三种调制的星座图（含噪 vs 干净）
- `figures/fig5_strategy.png` — 三种训练策略柱状对比

### 局限与下一步

1. 只做了单载波、单径 AWGN + 三类损伤，未考虑多径衰落与频选信道
2. 特征需手工设计，下一步可用 1D-CNN 直接从 IQ 序列端到端学习
3. 类别只覆盖三类，扩展到 8PSK / 64QAM 需要重新做特征筛选

---

## 项目二：时序传感数据异常检测

传感器采集的温度 / 湿度 / 电流数据流，做实时异常检测并可视化告警。
设计上支持三种数据源：内置仿真（默认）、CSV 文件、STM32 串口实时读取。

### 运行

```bash
python detect.py                    # 仿真数据
python detect.py --source csv --csv data.csv
python detect.py --source serial --port COM3
python plot_detect.py               # 生成图表
```

### 方法

- 滑窗：窗长 32 点，步进 8 点，窗内只要有异常点即标为异常
- 每窗提取 21 维统计特征（均值/标准差/极值/平均绝对差分/端点漂移/最大偏离 × 3 通道）
- 对比两种方法：3-sigma 阈值（工业现场常用土办法）vs 孤立森林

### 实测结果

| 方法 | 精确率 | 召回率 | F1 |
|---|---|---|---|
| 3-sigma 基线 | 0.991 | 0.727 | 0.839 |
| 孤立森林（contamination=0.06） | 1.000 | 0.315 | 0.479 |
| 孤立森林（调参后 0.15） | 0.973 | 0.762 | **0.855** |

> 第二个有价值的发现：孤立森林初次的召回率只有 0.315，远差于 3-sigma。
> 原因是 contamination 参数拍脑袋设成 0.06，而真实异常窗口占比是 19.1%，
> 模型被强制约束在只标 6% 的样本为异常。扫参到 0.15 后 F1 从 0.479 升到 0.855。

### 图表

- `figures/fig6_detection_timeline.png` — 原始信号 + 真实故障段 + 两种方法告警对比
- `figures/fig7_contamination_sweep.png` — contamination 敏感性曲线

---

## 目录结构

```
amc_project/
├── amc.py                  # 项目一主程序：信号生成 + 特征 + 分类
├── augment.py              # 项目一：数据增强 / 域适应对比
├── plot.py                 # 项目一：出图
├── detect.py               # 项目二主程序：滑窗 + 异常检测
├── plot_detect.py          # 项目二：出图
├── app.py                  # 项目二：Gradio 可视化界面
├── results.json/.txt       # 项目一实测结果
├── augment_results.json    # 数据增强实测结果
├── detect_results.json/.txt# 项目二实测结果
└── figures/                # 全部图表
```

## 参考

- 累积量特征参考：Dobre O A, et al. *Survey of automatic modulation classification techniques.*
- 根升余弦滤波器：通信原理教材标准实现，β=0.35 为 3GPP 常用值
