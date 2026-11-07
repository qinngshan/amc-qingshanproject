"""异常检测结果可视化."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

for cand in ["Microsoft YaHei", "SimHei"]:
    try:
        from matplotlib import font_manager
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

d = np.load(os.path.join(OUT, "data.npz"))
X = d["X"]
label = d["label"]
centers = d["centers"]
pred_if = d["pred_if"]
pred_3s = d["pred_3s"]


def fig6_timeline():
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 6.4), sharex=True)

    # 上图：原始信号 + 真实故障区间
    ax = axes[0]
    ax.plot(X[:, 0], X[:, 1], lw=0.8, color="#1f3a5f", label="温度 ℃")
    ax.plot(X[:, 0], X[:, 3] * 20 - 20, lw=0.6, color="#e67e22", alpha=0.7,
            label="电流 A（缩放显示）")
    ymin, ymax = ax.get_ylim()
    ax.fill_between(X[:, 0], ymin, ymax, where=label > 0, color="#c0392b",
                    alpha=0.18, label="真实故障段")
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("数值")
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.set_title("时序异常检测：从原始信号到模型告警", fontsize=13,
                 fontweight="bold")
    ax.grid(alpha=0.25, ls=":")

    # 中图：3-sigma 告警
    ax = axes[1]
    ax.vlines(centers, 0, pred_3s, color="#7f8c8d", lw=1.2,
              label="3-sigma 告警")
    ax.vlines(centers, 0, -label[centers] * 0.8, color="#c0392b", lw=1.2,
              alpha=0.6, label="真实异常（向下）")
    ax.set_ylim(-1.2, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["", "3σ"])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, ls=":")

    # 下图：孤立森林告警
    ax = axes[2]
    ax.vlines(centers, 0, pred_if, color="#27ae60", lw=1.2,
              label="孤立森林告警")
    ax.vlines(centers, 0, -label[centers] * 0.8, color="#c0392b", lw=1.2,
              alpha=0.6, label="真实异常（向下）")
    ax.set_ylim(-1.2, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["", "IF"])
    ax.set_xlabel("采样点")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, ls=":")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_detection_timeline.png"))
    plt.close(fig)


def fig7_sweep():
    rep = json.load(open(os.path.join(OUT, "detect_results.json"),
                         encoding="utf-8"))
    sw = rep.get("iforest_contamination_sweep", {})
    if not sw:
        return
    xs = [float(k) for k in sw]
    f1 = [sw[k]["f1"] for k in sw]
    p = [sw[k]["precision"] for k in sw]
    r = [sw[k]["recall"] for k in sw]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(xs, f1, "o-", lw=2, color="#1f3a5f", label="F1")
    ax.plot(xs, p, "s--", lw=1.4, color="#2980b9", label="精确率")
    ax.plot(xs, r, "^--", lw=1.4, color="#c0392b", label="召回率")
    best_c = rep.get("iforest_best_contamination")
    if best_c:
        ax.axvline(float(best_c), ls=":", color="#27ae60")
        ax.text(float(best_c) + 0.004, 0.25,
                f"最优 contamination={best_c}", fontsize=9, color="#27ae60")
    ax.set_xlabel("contamination（假定的异常比例）")
    ax.set_ylabel("指标值")
    ax.set_title("孤立森林调参：为什么 contamination 不能拍脑袋",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_contamination_sweep.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig6_timeline()
    print("ok fig6")
    fig7_sweep()
    print("ok fig7")
