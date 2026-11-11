# -*- coding: utf-8 -*-
"""
通信网络 KPI 数据分析实践：
模拟 50 个基站小区的 KPI（RSRP / SINR / 掉线率），掺入弱覆盖/干扰/高掉线三类问题小区，
对比「朴素 3-sigma」与「MAD 鲁棒阈值」识别问题小区，输出清单与优化建议。
"""
import numpy as np
import json

RNG = np.random.default_rng(42)
N_CELLS = 50

# ---------- 1. 生成正常小区 KPI ----------
rsrp = RNG.normal(-90, 3, N_CELLS)     # RSRP dBm，正常 -90 左右
sinr = RNG.normal(15, 3, N_CELLS)      # SINR dB，正常 15 左右
drop = RNG.uniform(0.5, 2.0, N_CELLS)  # 掉线率 %，正常 0.5~2

# ---------- 2. 掺入三类问题小区（真实标签，用于评估） ----------
truth = {}  # cell_id -> 问题类型
def inject(indices, kind, mutate):
    for i in indices:
        mutate(i)
        truth[int(i)] = kind

# 弱覆盖：RSRP 异常低
weak_idx = RNG.choice(N_CELLS, 4, replace=False)
inject(weak_idx, "弱覆盖", lambda i: rsrp.__setitem__(i, RNG.normal(-110, 3)))
# 干扰：SINR 异常低
interf_idx = RNG.choice(N_CELLS, 4, replace=False)
inject(interf_idx, "干扰", lambda i: sinr.__setitem__(i, RNG.normal(1, 1)))
# 高掉线：掉线率异常高
drop_idx = RNG.choice(N_CELLS, 4, replace=False)
inject(drop_idx, "高掉线", lambda i: drop.__setitem__(i, RNG.uniform(6, 12)))

# ---------- 3. 朴素 3-sigma（均值±3倍标准差） ----------
def naive_3sigma(x, lower):
    m, s = x.mean(), x.std()
    thr = m - 3 * s if lower else m + 3 * s
    return (x < thr) if lower else (x > thr)

weak_hit_naive = set(np.where(naive_3sigma(rsrp, True))[0].tolist())
interf_hit_naive = set(np.where(naive_3sigma(sinr, True))[0].tolist())
drop_hit_naive = set(np.where(naive_3sigma(drop, False))[0].tolist())
naive_hits = weak_hit_naive | interf_hit_naive | drop_hit_naive

# ---------- 4. MAD 鲁棒阈值（中位数 ± k*1.4826*MAD） ----------
def mad_threshold(x, lower, k=3.0):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    thr = med - k * sigma if lower else med + k * sigma
    return (x < thr) if lower else (x > thr)

weak_hit_mad = set(np.where(mad_threshold(rsrp, True))[0].tolist())
interf_hit_mad = set(np.where(mad_threshold(sinr, True))[0].tolist())
drop_hit_mad = set(np.where(mad_threshold(drop, False))[0].tolist())
mad_hits = weak_hit_mad | interf_hit_mad | drop_hit_mad

# ---------- 5. 评估 ----------
truth_set = set(truth.keys())
def metrics(hits):
    tp = len(hits & truth_set)
    fp = len(hits - truth_set)
    fn = len(truth_set - hits)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=round(p, 3), recall=round(r, 3), f1=round(f1, 3))

naive_m = metrics(naive_hits)
mad_m = metrics(mad_hits)

print("=" * 56)
print("通信网络 KPI 数据分析 —— 实测结果")
print("=" * 56)
print(f"小区总数: {N_CELLS}  问题小区(真实): {len(truth_set)} (占比 {len(truth_set)/N_CELLS*100:.1f}%)")
print()
print(f"朴素 3-sigma  识别: {len(naive_hits)} 个  TP={naive_m['tp']} FP={naive_m['fp']} FN={naive_m['fn']}  P={naive_m['precision']} R={naive_m['recall']} F1={naive_m['f1']}")
print(f"MAD 鲁棒阈值  识别: {len(mad_hits)} 个  TP={mad_m['tp']} FP={mad_m['fp']} FN={mad_m['fn']}  P={mad_m['precision']} R={mad_m['recall']} F1={mad_m['f1']}")
print()
print("被朴素 3-sigma 漏掉、MAD 识别出来的问题小区:")
missed = truth_set - naive_hits
for cid in sorted(missed):
    kind = truth[cid]
    if cid in mad_hits:
        print(f"  小区{cid:02d} [{kind}]: RSRP={rsrp[cid]:.1f} SINR={sinr[cid]:.1f} 掉线率={drop[cid]:.2f}%  → MAD 命中")
print()
print("优化建议清单（按类型）:")
for kind in sorted(set(truth.values())):
    ids = sorted(c for c, k in truth.items() if k == kind)
    hint = {"弱覆盖": "增补站点/调整天线下倾角，覆盖补盲",
            "干扰": "排查同频干扰源/调整频率规划与功率",
            "高掉线": "邻区关系与切换参数优化，降切换失败"}[kind]
    print(f"  {kind} ({len(ids)} 个): 小区 {ids}  →  {hint}")

# ---------- 6. 保存结果 ----------
out = {
    "n_cells": N_CELLS,
    "truth_count": len(truth_set),
    "truth_ratio": round(len(truth_set) / N_CELLS, 3),
    "naive_3sigma": naive_m,
    "mad_robust": mad_m,
    "naive_missed_then_mad_hit": sorted(truth_set - naive_hits),
}
with open("kpi_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n结果已保存 kpi_results.json")
