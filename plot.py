"""
结果可视化: 画出 SNR-准确率曲线、混淆矩阵、特征重要性、星座图.
输出到 figures/ 目录, 可直接放进 GitHub README 和面试 PPT.

用法: python plot.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import amc

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# 中文字体
for cand in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


def fig1_snr_curve():
    """SNR - 准确率曲线 (三种训练策略对比)."""
    rep = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    aug = json.load(open(os.path.join(OUT, "augment_results.json"),
                         encoding="utf-8"))
    snrs = rep["snrs"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    best = rep["best_model"]
    ys = [rep["by_snr"][best][str(s)] * 100 for s in snrs]
    ax.plot(snrs, ys, "o-", lw=2, ms=6, color="#1f3a5f",
            label=f"{best}（理想信道训练，理想信道测试）")

    # 域适应前后的对比 (取随机森林, 两组数据都有)
    a = aug["strategy"]["A. 仅理想信道训练"]["随机森林"]
    c = aug["strategy"]["C. 理想+损伤混合训练"]["随机森林"]
    ax.axhline(a["含损伤信道"], ls="--", lw=1.5, color="#c0392b",
               label=f"未见损伤时跨信道  {a['含损伤信道']:.1f}%")
    ax.axhline(c["含损伤信道"], ls="-.", lw=1.5, color="#27ae60",
               label=f"数据增强后跨信道  {c['含损伤信道']:.1f}%")

    ax.set_xlabel("信噪比 SNR (dB)", fontsize=11)
    ax.set_ylabel("识别准确率 (%)", fontsize=11)
    ax.set_title("调制方式识别：准确率随信噪比变化", fontsize=13, fontweight="bold")
    ax.set_ylim(50, 103)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_snr_accuracy.png"))
    plt.close(fig)


def fig2_confusion():
    rep = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    cm = np.array(rep["confusion_matrix"], dtype=float)
    ratio = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(ratio, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(amc.MODS)))
    ax.set_yticks(range(len(amc.MODS)))
    ax.set_xticklabels(amc.MODS)
    ax.set_yticklabels(amc.MODS)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵（随机森林，理想信道）", fontsize=12, fontweight="bold")
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            ax.text(j, i, f"{ratio[i, j]:.1f}%", ha="center", va="center",
                    color="white" if ratio[i, j] > 50 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="百分比 (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_confusion.png"))
    plt.close(fig)


def fig3_importance():
    rep = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    names = ["C40", "C42", "C63", "|C41|", "幅度标准差", "峰均比",
             "相位标准差", "频谱峭度"]
    imp = rep["feature_importance"]
    order = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh([names[i] for i in order], [imp[i] for i in order],
            color="#1f3a5f", alpha=0.85)
    ax.set_xlabel("特征重要性（Gini 增益）")
    ax.set_title("哪些特征真正在起作用", fontsize=12, fontweight="bold")
    for i, v in enumerate([imp[i] for i in order]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_importance.png"))
    plt.close(fig)


def fig4_constellation():
    """三类信号的星座图 (理想 vs 含噪声), 这是面试最能讲的一张图."""
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.0))
    for j, mod in enumerate(amc.MODS):
        for i, snr in enumerate([20, 2]):
            x = amc.make_signal(mod, n_sym=600, snr=snr)
            axes[i, j].plot(x.real[::4], x.imag[::4], ".", ms=2,
                            alpha=0.5, color="#1f3a5f")
            axes[i, j].set_title(f"{mod}  SNR={snr}dB", fontsize=10)
            axes[i, j].set_xlim(-2, 2)
            axes[i, j].set_ylim(-2, 2)
            axes[i, j].grid(alpha=0.25, ls=":")
            axes[i, j].set_aspect("equal")
    fig.suptitle("三种调制方式的星座图：噪声如何破坏可分性",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_constellation.png"))
    plt.close(fig)


def fig5_strategy():
    """三种训练策略的柱状对比."""
    aug = json.load(open(os.path.join(OUT, "augment_results.json"),
                         encoding="utf-8"))
    labels, ideal, imp = [], [], []
    for sname, row in aug["strategy"].items():
        best = max(row.items(), key=lambda kv: kv[1]["含损伤信道"])
        labels.append(sname.split(". ")[1])
        ideal.append(best[1]["理想信道"])
        imp.append(best[1]["含损伤信道"])

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    b1 = ax.bar(x - w / 2, ideal, w, label="理想信道", color="#8fa8c8")
    b2 = ax.bar(x + w / 2, imp, w, label="含损伤信道", color="#c0392b")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                    f"{b.get_height():.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("准确率 (%)")
    ax.set_ylim(0, 108)
    ax.set_title("数据增强如何补回跨信道性能损失",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_strategy.png"))
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig1_snr_curve, fig2_confusion, fig3_importance,
              fig4_constellation, fig5_strategy):
        f()
        print("ok:", f.__name__)
    print("figures ->", FIG)
