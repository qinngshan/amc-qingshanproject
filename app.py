"""
Gradio 可视化界面 —— 把异常检测模型包成可演示的小工具.
面试时可以直接打开给面试官看（本地跑，不需要联网）.

运行: python app.py    然后浏览器打开 http://127.0.0.1:7860
"""
import numpy as np

try:
    import gradio as gr
    HAS_GR = True
except ImportError:
    HAS_GR = False

import detect as D


def run_detection(fault_ratio, win, contamination):
    """界面回调：生成数据 -> 检测 -> 返回三张图 + 指标文字."""
    global _WIN
    import importlib
    D.WIN = int(win)
    importlib.reload  # noqa
    X, label, kinds = D.gen_sim_data(n=4000, fault_ratio=fault_ratio)
    F, centers = D.window_features(X)
    yw = D.label_windows(label, centers)

    m3 = D.ThreeSigma(k=3.0).fit(F)
    p3 = m3.predict(F)
    mif = D.IForest(contamination=contamination).fit(F)
    pif = mif.predict(F)

    r3 = D.evaluate(yw, p3)
    rif = D.evaluate(yw, pif)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for cand in ["Microsoft YaHei", "SimHei"]:
        try:
            font_manager.findfont(cand, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [cand]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(X[:, 0], X[:, 1], lw=0.8, color="#1f3a5f", label="温度")
    ymin, ymax = axes[0].get_ylim()
    axes[0].fill_between(X[:, 0], ymin, ymax, where=label > 0,
                         color="#c0392b", alpha=0.18, label="真实故障")
    axes[0].set_ylim(ymin, ymax)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("温度 ℃")
    axes[0].set_title("原始信号与真实故障段")

    axes[1].vlines(centers, 0, p3, color="#7f8c8d", lw=1.2)
    axes[1].vlines(centers, 0, -yw * 0.8, color="#c0392b", lw=1.2, alpha=0.6)
    axes[1].set_ylabel("3σ 告警")
    axes[1].set_ylim(-1.2, 1.3)

    axes[2].vlines(centers, 0, pif, color="#27ae60", lw=1.2)
    axes[2].vlines(centers, 0, -yw * 0.8, color="#c0392b", lw=1.2, alpha=0.6)
    axes[2].set_ylabel("孤立森林")
    axes[2].set_xlabel("采样点")
    axes[2].set_ylim(-1.2, 1.3)
    fig.tight_layout()

    txt = (f"异常窗口占比（真实）：{yw.mean() * 100:.1f}%\n\n"
           f"3-sigma 基线     精确率 {r3['precision']:.3f}  "
           f"召回率 {r3['recall']:.3f}  F1 {r3['f1']:.3f}\n"
           f"孤立森林          精确率 {rif['precision']:.3f}  "
           f"召回率 {rif['recall']:.3f}  F1 {rif['f1']:.3f}\n\n"
           f"参数 contamination 冻结为 {contamination:.2f}，"
           f"与真实占比 {yw.mean():.2f} 的差距决定召回率高低。")
    return fig, txt


def build():
    with gr.Blocks(title="传感数据异常检测演示") as demo:
        gr.Markdown("## 时序传感数据异常检测演示\n"
                    "滑窗特征 + 3-sigma / 孤立森林对比。"
                    "调节参数可观察 contamination 如何影响召回率。")
        with gr.Row():
            fault = gr.Slider(0.02, 0.20, value=0.08, step=0.01,
                              label="故障段占比（模拟参数）")
            win = gr.Slider(16, 64, value=32, step=8, label="滑窗长度")
            cont = gr.Slider(0.02, 0.25, value=0.08, step=0.01,
                             label="contamination（模型假定的异常比例）")
        btn = gr.Button("运行检测", variant="primary")
        with gr.Row():
            plot = gr.Plot(label="三通道时序与告警")
            metric = gr.Textbox(label="评估指标", lines=8)
        btn.click(run_detection, [fault, win, cont], [plot, metric])
        gr.Markdown("提示：把 contamination 调到接近真实故障占比，"
                    "观察孤立森林的 F1 如何变化 —— 这就是调参的意义。")
    return demo


if __name__ == "__main__":
    if not HAS_GR:
        print("请先安装 gradio: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gradio")
    else:
        build().launch(server_name="127.0.0.1", server_port=7860)
