"""
调制方式自动识别 (AMC) —— 通信 + 机器学习交叉项目
====================================================

流程:
  1. 生成 BPSK / QPSK / 16QAM 三类信号的基带 IQ 序列 (含脉冲成型)
  2. 叠加 AWGN 高斯白噪声, 扫 SNR 从 -2dB 到 16dB
  3. 提取 8 维特征: 高阶累积量 + 瞬时统计量 + 频谱特征
  4. 训练 4 个模型: KNN / SVM / 随机森林 / 1D-CNN (PyTorch 可选)
  5. 输出: 各 SNR 下的准确率曲线、混淆矩阵、特征重要性

真实运行结果写进 results.txt, 简历里的数字用实测值。

用法:
    python amc.py              # 跑完整实验
    python amc.py --quick      # 减少样本量, 快速验证流程
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RNG = np.random.default_rng(20260101)
MODS = ["BPSK", "QPSK", "16QAM"]
SNRS = [-2, 0, 2, 4, 6, 8, 10, 12, 16]

# ---------------------------------------------------------------- 1. 信号生成
def rrc_filter(beta=0.35, span=8, sps=4):
    """根升余弦脉冲成型滤波器 (通信原理课上的标准做法)."""
    t = np.arange(-span / 2, span / 2 + 1e-9, 1 / sps)
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(4 * beta * ti) - 1) < 1e-9:
            h[i] = beta / np.sqrt(2) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta))
                    + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (
                np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def gen_symbols(mod, n):
    if mod == "BPSK":
        return np.exp(1j * np.pi * RNG.integers(0, 2, n))
    if mod == "QPSK":
        return np.exp(1j * (np.pi / 4 + np.pi / 2 * RNG.integers(0, 4, n)))
    if mod == "16QAM":
        lvl = np.array([-3, -1, 1, 3])
        a = RNG.choice(lvl, n)
        b = RNG.choice(lvl, n)
        return (a + 1j * b) / np.sqrt(10)
    raise ValueError(mod)


def add_awgn(x, snr_db):
    """按符号功率加噪声, 返回功率归一化后的信号."""
    p = np.mean(np.abs(x) ** 2)
    npow = p / (10 ** (snr_db / 10))
    n = np.sqrt(npow / 2) * (RNG.standard_normal(x.size)
                            + 1j * RNG.standard_normal(x.size))
    y = x + n
    return y / np.sqrt(np.mean(np.abs(y) ** 2))


def make_signal(mod, n_sym=1024, snr=10, sps=4, cfo=0.0, pn=0.0,
                timing_err=0.0):
    """生成一条调制信号: 符号 -> 上采样 -> 脉冲成型 -> 加损伤 -> 加噪.

    真实信道的三类损伤 (面试会问「你的仿真和实际差在哪」):
      cfo        载波频偏 (Hz/符号率), 造成星座图旋转 -> 相位特征被破坏
      pn         相位噪声标准差 (rad), 振荡器抖动
      timing_err 定时误差 (采样点偏移比例), 模拟非理想采样
    """
    syms = gen_symbols(mod, n_sym)
    up = np.zeros(n_sym * sps, dtype=complex)
    up[::sps] = syms
    h = rrc_filter(sps=sps)
    base = np.convolve(up, h, mode="same")
    n = base.size

    if timing_err:
        idx = np.arange(n) + timing_err
        base = interp_complex(base, idx)

    if cfo:
        t = np.arange(n) / sps                 # 以符号率为单位的时间
        base = base * np.exp(2j * np.pi * cfo * t)

    if pn:
        base = base * np.exp(1j * pn * np.cumsum(RNG.standard_normal(n)))

    return add_awgn(base, snr)


def interp_complex(x, idx):
    """复数序列线性插值 (numpy.interp 只支持实数)."""
    i0 = np.floor(idx).astype(int).clip(0, x.size - 1)
    i1 = (i0 + 1).clip(0, x.size - 1)
    w = idx - i0
    return x[i0] * (1 - w) + x[i1] * w


# ---------------------------------------------------------------- 2. 特征提取
def _moments(x, k):
    return np.mean(x ** k)


def extract_features(x):
    """8 维特征: 4 个高阶累积量 + 3 个瞬时统计量 + 1 个频谱特征.

    为什么这么选 (面试必问):
      - BPSK/QPSK 都是恒包络 PSK, 四阶累积量 C42 在两者间差异大 (符号对称性不同)
      - 16QAM 幅度非恒定, 瞬时幅度方差 / 峰均比能把它和 PSK 类分开
      - 归一化消除 SNR 对绝对幅度的影响 (否则模型学的是信噪比不是调制方式)
    """
    x = x - np.mean(x)          # 去 DC
    p = np.mean(np.abs(x) ** 2)
    x = x / np.sqrt(p)          # 单位功率归一化 (关键!)

    m20 = _moments(x, 2)
    m40 = _moments(x, 4)
    m41 = np.mean(x ** 3 * np.conj(x))       # C41: 三阶乘共轭
    m42 = np.mean(np.abs(x) ** 4)
    m63 = np.mean(np.abs(x) ** 6)
    c40 = m40 - 3 * m20 ** 2
    c42 = m42 - abs(m20) ** 2 - 2 * np.mean(np.abs(x) ** 2) ** 2
    c63 = m63 - 9 * m42 * np.mean(np.abs(x) ** 2) + 12 * np.mean(
        np.abs(x) ** 2) ** 3

    amp = np.abs(x)
    phase = np.unwrap(np.angle(x))
    amp_std = np.std(amp)
    papr = np.max(amp) ** 2 / np.mean(amp ** 2)
    phase_std = np.std(np.diff(phase))

    spec = np.abs(np.fft.fftshift(np.fft.fft(x)))
    spec = spec / (np.sum(spec) + 1e-12)
    spec_kurt = float(np.sum(((np.arange(spec.size) - spec.size / 2) ** 4)
                             * spec))
    spec_kurt = np.log1p(spec_kurt)

    return np.array([np.real(c40), np.real(c42), np.real(c63),
                     abs(m41) if np.isfinite(abs(m41)) else 0.0,
                     amp_std, papr, phase_std, spec_kurt], dtype=np.float64)


# ---------------------------------------------------------------- 3. 数据集
def build_dataset(n_per_class, snrs, n_sym=1024, impair=False):
    """impair=True 时加入载波频偏/相位噪声/定时误差, 模拟真实接收机."""
    X, y, s = [], [], []
    for mod_i, mod in enumerate(MODS):
        for snr in snrs:
            for _ in range(n_per_class):
                if impair:
                    kw = dict(cfo=RNG.uniform(-0.002, 0.002),
                              pn=RNG.uniform(0.005, 0.03),
                              timing_err=RNG.uniform(-0.3, 0.3))
                else:
                    kw = {}
                x = make_signal(mod, n_sym=n_sym, snr=snr, **kw)
                X.append(extract_features(x))
                y.append(mod_i)
                s.append(snr)
    return np.array(X), np.array(y), np.array(s)


# ---------------------------------------------------------------- 4. 主实验
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    n_per = 40 if args.quick else 150
    snrs = [-2, 0, 4, 8, 12, 16] if args.quick else SNRS

    t0 = time.time()
    print(f"[1/4] 生成数据: {len(MODS)} 类 x {len(snrs)} 个 SNR x {n_per} 条 "
          f"x 2 种信道 (理想 / 含损伤)")
    Xa, ya, sa = build_dataset(n_per, snrs, impair=False)
    Xb, yb, sb = build_dataset(n_per, snrs, impair=True)
    print(f"      样本数 = {len(Xa)} + {len(Xb)}, 特征维度 = {Xa.shape[1]}  "
          f"({time.time() - t0:.1f}s)")

    print("[2/4] 划分训练/测试集 (分层抽样)")
    Xtr, Xte, ytr, yte, str_, ste = train_test_split(
        Xa, ya, sa, test_size=0.3, random_state=42, stratify=ya)
    # 跨信道测试: 训练用理想信道, 测试用含损伤信道 -> 检验泛化能力
    Xcx, ycx, scx = Xb, yb, sb

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s, Xcx_s = sc.transform(Xtr), sc.transform(Xte), sc.transform(Xcx)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale"),
        "随机森林": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1),
    }

    print("[3/4] 训练与评估")
    report = {"n_samples": int(len(Xa) + len(Xb)),
              "n_features": int(Xa.shape[1]),
              "snrs": snrs, "models": {}, "by_snr": {}, "cross_channel": {}}

    for name, clf in models.items():
        clf.fit(Xtr_s, ytr)
        pred = clf.predict(Xte_s)
        acc = accuracy_score(yte, pred)
        report["models"][name] = float(acc)
        print(f"      {name:6s} 理想信道 = {acc * 100:.2f}%")

        per_snr = {}
        for snr in sorted(set(ste)):
            m = ste == snr
            if m.sum() == 0:
                continue
            per_snr[str(snr)] = float(accuracy_score(yte[m], pred[m]))
        report["by_snr"][name] = per_snr

        # 跨信道泛化
        acc_cx = accuracy_score(ycx, clf.predict(Xcx_s))
        report["cross_channel"][name] = float(acc_cx)
        print(f"             跨信道(含频偏/相噪/定时误差) = {acc_cx * 100:.2f}%")

        if name == "随机森林":
            report["confusion_matrix"] = confusion_matrix(yte, pred).tolist()
            report["feature_importance"] = clf.feature_importances_.tolist()

    hi = ste >= 8
    if hi.sum():
        best = max(report["models"], key=report["models"].get)
        pred_best = models[best].predict(Xte_s)
        report["high_snr_acc"] = float(accuracy_score(yte[hi], pred_best[hi]))
        report["best_model"] = best

    print("[4/4] 保存结果")
    out = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = ["调制方式自动识别 —— 实测结果", "=" * 40,
             f"样本总数: {report['n_samples']}  特征维度: {report['n_features']}",
             "", "各模型总体准确率:"]
    for k, v in report["models"].items():
        lines.append(f"  {k}: {v * 100:.2f}%")
    if "high_snr_acc" in report:
        lines.append(f"\nSNR>=8dB 时最优模型 ({report['best_model']}) "
                     f"准确率: {report['high_snr_acc'] * 100:.2f}%")
    lines.append("\n跨信道泛化 (训练=理想信道, 测试=含频偏/相噪/定时误差):")
    for name, v in report["cross_channel"].items():
        lines.append(f"  {name}: {v * 100:.2f}%")
    lines += ["", "分 SNR 准确率:"]
    for name, d in report["by_snr"].items():
        lines.append(f"  {name}: " + "  ".join(
            f"{k}dB={v * 100:.1f}%" for k, v in d.items()))
    lines += ["", "随机森林特征重要性:"]
    feat_names = ["C40", "C42", "C63", "|C41|", "幅度标准差", "峰均比",
                  "相位标准差", "频谱峭度"]
    for n, v in zip(feat_names, report.get("feature_importance", [])):
        lines.append(f"  {n}: {v:.4f}")
    with open(os.path.join(out, "results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
