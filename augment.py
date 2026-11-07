"""
数据增强 + 域适应实验
=====================
问题: 用理想信道训练的模型, 遇到含载波频偏/相位噪声/定时误差的信号,
      准确率从 98% 掉到 63%(KNN)。
思路: 训练集里混入带损伤的样本 (数据增强), 让模型见过真实损伤。
      分别测试"全随机损伤"和"损伤强度分层"两种增强策略。

输出: augment_results.txt —— 简历/面试用的对比数字。
"""
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import amc
import importlib
importlib.reload(amc)

OUT = os.path.dirname(os.path.abspath(__file__))


def build_clean(n_per, snrs):
    X, y, s = [], [], []
    for mi, mod in enumerate(amc.MODS):
        for snr in snrs:
            for _ in range(n_per):
                X.append(amc.extract_features(
                    amc.make_signal(mod, snr=snr)))
                y.append(mi)
                s.append(snr)
    return np.array(X), np.array(y), np.array(s)


def build_impaired(n_per, snrs, cfo_rng, pn_rng, te_rng):
    """按给定损伤区间生成样本."""
    X, y, s = [], [], []
    for mi, mod in enumerate(amc.MODS):
        for snr in snrs:
            for _ in range(n_per):
                kw = dict(cfo=amc.RNG.uniform(*cfo_rng),
                          pn=amc.RNG.uniform(*pn_rng),
                          timing_err=amc.RNG.uniform(*te_rng))
                X.append(amc.extract_features(
                    amc.make_signal(mod, snr=snr, **kw)))
                y.append(mi)
                s.append(snr)
    return np.array(X), np.array(y), np.array(s)


def main():
    n_per = 130
    snrs = amc.SNRS

    print("[1/4] 生成理想信道数据")
    Xc, yc, sc = build_clean(n_per, snrs)
    print(f"      {len(Xc)} 条")

    print("[2/4] 生成含损伤数据 (真实接收机场景)")
    Xi, yi, si = build_impaired(n_per, snrs,
                                cfo_rng=(-0.002, 0.002),
                                pn_rng=(0.005, 0.03),
                                te_rng=(-0.3, 0.3))
    print(f"      {len(Xi)} 条")

    # 划分: 同一套测试集, 保证可比
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        Xc, yc, test_size=0.3, random_state=42, stratify=yc)
    Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(
        Xi, yi, test_size=0.3, random_state=42, stratify=yi)

    models = {
        "KNN": lambda: KNeighborsClassifier(n_neighbors=7),
        "SVM": lambda: SVC(kernel="rbf", C=10, gamma="scale"),
        "随机森林": lambda: RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1),
    }

    report = {"n_train_clean": int(len(Xc_tr)),
              "n_train_impaired": int(len(Xi_tr)),
              "strategy": {}}

    print("[3/4] 三种训练策略对比")
    strategies = [
        ("A. 仅理想信道训练", Xc_tr, yc_tr),
        ("B. 仅含损伤训练", Xi_tr, yi_tr),
        ("C. 理想+损伤混合训练", np.vstack([Xc_tr, Xi_tr]),
         np.concatenate([yc_tr, yi_tr])),
    ]
    for sname, Xtr, ytr in strategies:
        sc_ = StandardScaler().fit(Xtr)
        row = {}
        for mname, factory in models.items():
            clf = factory().fit(sc_.transform(Xtr), ytr)
            acc_clean = accuracy_score(yc_te, clf.predict(sc_.transform(Xc_te)))
            acc_imp = accuracy_score(yi_te, clf.predict(sc_.transform(Xi_te)))
            row[mname] = {"理想信道": round(acc_clean * 100, 2),
                          "含损伤信道": round(acc_imp * 100, 2)}
            print(f"      {sname:18s} {mname:6s} 理想={acc_clean * 100:5.2f}%  "
                  f"含损伤={acc_imp * 100:5.2f}%")
        report["strategy"][sname] = row

    print("[4/4] 保存")
    with open(os.path.join(OUT, "augment_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = ["数据增强 / 域适应实验 —— 实测结果", "=" * 46, "",
             f"训练集: 理想 {report['n_train_clean']} 条 / "
             f"含损伤 {report['n_train_impaired']} 条", ""]
    for sname, row in report["strategy"].items():
        lines.append(sname)
        for m, d in row.items():
            lines.append(f"  {m:6s} 理想信道 {d['理想信道']:6.2f}%   "
                         f"含损伤信道 {d['含损伤信道']:6.2f}%")
        lines.append("")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "augment_results.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
