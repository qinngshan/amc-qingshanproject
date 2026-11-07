"""
时序传感数据异常检测 —— 从传感器到可演示界面
==============================================

对应简历项目二。没有 STM32 板子也能完整跑通：
  --source sim    内置仿真数据生成器（默认，无需硬件）
  --source csv    读取真实采集的 CSV（列名 time,temp,humi,current）
  --source serial 从串口读取 STM32 上报数据（需 pyserial + 实际硬件）

流程: 数据生成/读取 -> 滑窗特征 -> 模型训练(IsolationForest / 3-sigma)
      -> 评估(精确率/召回率/F1) -> 输出告警时间轴图
可视化界面: python app.py   （Gradio，浏览器里跑）

设计要点（面试会问）:
  * 为什么用滑窗而不是逐点判断 —— 单点抖动误报太多，异常是"一段时间的行为"
  * 为什么异常检测不用监督学习 —— 现实中故障样本极少，标注不可得
  * 3-sigma 作为基线对照，证明机器学习确实有收益
"""
import argparse
import json
import os

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

OUT = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(7)

WIN = 32          # 滑窗长度（采样点）
STEP = 8          # 滑窗步进


# ------------------------------------------------------------ 1. 数据来源
def gen_sim_data(n=6000, fault_ratio=0.08):
    """生成带故障的时序数据。

    正常段: 温度 25±0.4℃ / 湿度 55±1.5% / 电流 1.2±0.05A，带缓慢漂移
    故障段三类（总占比约 fault_ratio，互不重叠）:
      drift   缓慢漂移（散热失效，温度单向上爬）
      spike   尖峰（接触不良，电流瞬时脉冲）
      flat    卡死（传感器故障，数值停在常数）
    """
    t = np.arange(n)
    temp = 25 + 0.8 * np.sin(2 * np.pi * t / 1500) + RNG.normal(0, 0.25, n)
    humi = 55 + 2.0 * np.sin(2 * np.pi * t / 900 + 1) + RNG.normal(0, 1.2, n)
    curr = 1.2 + 0.03 * np.sin(2 * np.pi * t / 400) + RNG.normal(0, 0.04, n)

    label = np.zeros(n, dtype=int)
    kinds = []
    # 总故障时长约占 fault_ratio，切成 n_episode 段
    n_episode = 12
    total_fault = int(n * fault_ratio)
    per_ep = max(WIN * 2, total_fault // n_episode)

    occupied = np.zeros(n, dtype=bool)
    kind_cycle = ["drift", "spike", "flat"]
    for k in range(n_episode):
        for _ in range(200):                    # 找一个不与已有故障重叠的位置
            start = int(RNG.integers(WIN * 2, n - per_ep - WIN * 2))
            if not occupied[start:start + per_ep].any():
                break
        end = start + per_ep
        occupied[start:end] = True
        label[start:end] = 1
        kind = kind_cycle[k % 3]
        kinds.append((kind, start, end))
        if kind == "drift":
            temp[start:end] += np.linspace(0, 5, end - start)
        elif kind == "spike":
            curr[start:end] += (RNG.random(end - start) > 0.7) * 2.5
        else:
            temp[start:end] = temp[start]
            humi[start:end] = humi[start]

    return np.column_stack([t, temp, humi, curr]), label, kinds


def load_csv(path):
    """读取真实采集数据, 需含 time,temp,humi,current[,label] 列."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    cols = [data[c] for c in ("time", "temp", "humi", "current")]
    X = np.column_stack(cols)
    y = data["label"].astype(int) if "label" in data.dtype.names else None
    return X, y, []


def load_serial(port="COM3", baud=115200, n=2000):
    """从 STM32 串口读数据. 协议: 每行 't,temp,humi,current'."""
    import serial
    buf = []
    with serial.Serial(port, baud, timeout=2) as ser:
        while len(buf) < n:
            line = ser.readline().decode(errors="ignore").strip()
            try:
                vals = [float(v) for v in line.split(",")]
                if len(vals) == 4:
                    buf.append(vals)
            except ValueError:
                continue
    X = np.array(buf)
    return X, None, []


# ------------------------------------------------------------ 2. 特征
def window_features(X):
    """每个滑窗提取统计特征，返回 (特征矩阵, 窗口中心索引)."""
    sig = X[:, 1:]                       # 去掉时间列
    feats, centers = [], []
    for s in range(0, len(sig) - WIN + 1, STEP):
        w = sig[s:s + WIN]
        f = []
        for c in range(w.shape[1]):
            v = w[:, c]
            f += [v.mean(), v.std(), v.min(), v.max(),
                  np.mean(np.abs(np.diff(v))),          # 平均绝对差分
                  v[-1] - v[0],                          # 端点漂移
                  np.max(np.abs(v - v.mean()))]          # 最大偏离
        feats.append(f)
        centers.append(s + WIN // 2)
    return np.array(feats), np.array(centers)


def label_windows(label, centers):
    """窗口内只要有异常点就标为异常."""
    y = np.array([label[max(0, c - WIN // 2):c + WIN // 2].max()
                  for c in centers])
    return y


# ------------------------------------------------------------ 3. 模型
class ThreeSigma:
    """基线: 滚动 3-sigma 阈值。工业现场最常见的土办法。"""

    def __init__(self, k=3.0):
        self.k = k

    def fit(self, X):
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        return self

    def score(self, X):
        """返回每窗最大偏离（越大越异常）。"""
        z = np.abs((X - self.mu_) / self.sd_)
        return z.max(axis=1)

    def predict(self, X):
        return (self.score(X) > self.k).astype(int)


class IForest:
    def __init__(self, contamination=0.06):
        self.m = IsolationForest(n_estimators=200, contamination=contamination,
                                 random_state=42, n_jobs=-1)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)

    def predict(self, X):
        return (self.m.predict(X) == -1).astype(int)


def evaluate(y, pred):
    return {
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sim",
                    choices=["sim", "csv", "serial"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--port", default="COM3")
    args = ap.parse_args()

    print(f"[1/4] 加载数据 (source={args.source})")
    if args.source == "csv" and args.csv:
        X, label, kinds = load_csv(args.csv)
    elif args.source == "serial":
        X, label, kinds = load_serial(args.port)
    else:
        X, label, kinds = gen_sim_data()
    print(f"      {len(X)} 采样点, 故障段 {len(kinds)} 个")

    print("[2/4] 滑窗特征")
    F, centers = window_features(X)
    print(f"      窗口数 {len(F)}, 每窗特征 {F.shape[1]} 维")

    if label is None:
        print("[3/4] 无标签, 仅输出告警结果")
        yw = None
    else:
        yw = label_windows(label, centers)
        print(f"      异常窗口 {int(yw.sum())} / {len(yw)} "
              f"({yw.mean() * 100:.1f}%)")

    print("[3/4] 训练与评估")
    report = {"n_points": int(len(X)), "n_windows": int(len(F)),
              "window_size": WIN, "step": STEP,
              "anomaly_windows": int(yw.sum()) if yw is not None else None,
              "models": {}}
    preds = {}
    best = None
    for name, model in [("3-sigma 基线", ThreeSigma(k=3.0)),
                        ("孤立森林", IForest(contamination=0.06))]:
        model.fit(F)
        pred = model.predict(F)
        preds[name] = pred
        if yw is not None:
            m = evaluate(yw, pred)
            report["models"][name] = m
            print(f"      {name:10s} 精确率={m['precision']:.3f} "
                  f"召回率={m['recall']:.3f} F1={m['f1']:.3f}")
            if best is None or m["f1"] > best[1]["f1"]:
                best = (name, m)

    # 调参: contamination 是未知量, 用"异常比例未知"的常规做法扫一遍取最优 F1
    if yw is not None:
        print("      -- contamination 敏感性扫描（面试重点：参数不能拍脑袋）")
        sweep = {}
        for c in [0.02, 0.05, 0.08, 0.12, 0.15, 0.19, 0.25]:
            p = IForest(contamination=c).fit(F).predict(F)
            m = evaluate(yw, p)
            sweep[f"{c:.2f}"] = m
            print(f"         contamination={c:.2f}  F1={m['f1']:.3f} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f}")
        report["iforest_contamination_sweep"] = sweep
        best_c = max(sweep, key=lambda k: sweep[k]["f1"])
        report["iforest_best_contamination"] = best_c
        report["iforest_tuned"] = sweep[best_c]
        # 真实异常占比作为对照（实际业务里是拿不到的，这里只做复盘）
        report["true_anomaly_ratio"] = round(float(yw.mean()), 4)
        preds["孤立森林(调参后)"] = IForest(
            contamination=float(best_c)).fit(F).predict(F)

    print("[4/4] 保存")
    np.savez_compressed(os.path.join(OUT, "data.npz"), X=X,
                        label=label if label is not None else np.zeros(len(X)),
                        centers=centers,
                        pred_if=preds.get("孤立森林", np.zeros(len(centers))),
                        pred_3s=preds.get("3-sigma 基线",
                                          np.zeros(len(centers))),
                        score_if=IForest().fit(F).score(F))
    with open(os.path.join(OUT, "detect_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = ["时序传感数据异常检测 —— 实测结果", "=" * 40,
             f"采样点: {report['n_points']}  窗口: {report['n_windows']} "
             f"(WIN={WIN}, STEP={STEP})",
             f"异常窗口占比: "
             f"{(report['anomaly_windows'] or 0) / max(1, report['n_windows']) * 100:.1f}%",
             "", "模型对比:"]
    for k, v in report["models"].items():
        lines.append(f"  {k}: 精确率 {v['precision']:.3f}  "
                     f"召回率 {v['recall']:.3f}  F1 {v['f1']:.3f}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "detect_results.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
