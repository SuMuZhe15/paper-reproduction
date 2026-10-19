
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 输出目录
OUTPUT_DIR = Path("results/visualization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 实验总体数据
methods = ["Standard", "CoT", "Act", "ReAct"]
em = [8.46, 0.00, 0.00, 10.87]
f1 = [19.95, 4.45, 4.76, 26.35]
runtime = [1.25, 2.45, 3.49, 5.94]
success_rate = [98.48, 98.26, 97.83, 92.83]

# 分类型 F1
types = ["Bridge", "Comparison"]
standard_f1 = [19.20, 23.23]
react_f1 = [28.60, 16.54]

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{filename}.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{filename}.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"已生成: {filename}.png / .pdf")


# 1. 总体 EM 和 F1
x = np.arange(len(methods))
width = 0.36

fig, ax = plt.subplots(figsize=(9, 5.5))
bars1 = ax.bar(x - width / 2, em, width, label="EM")
bars2 = ax.bar(x + width / 2, f1, width, label="F1")

ax.set_title("HotpotQA Overall Performance")
ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, 32)
ax.legend()
ax.grid(axis="y", alpha=0.25)

ax.bar_label(bars1, fmt="%.2f", padding=3)
ax.bar_label(bars2, fmt="%.2f", padding=3)

save_figure(fig, "01_overall_performance")


# 2. Bridge / Comparison 类型对比
x = np.arange(len(types))

fig, ax = plt.subplots(figsize=(8.5, 5.5))
bars1 = ax.bar(x - width / 2, standard_f1,
               width, label="Standard")
bars2 = ax.bar(x + width / 2, react_f1,
               width, label="ReAct")

ax.set_title("F1 by HotpotQA Question Type")
ax.set_ylabel("Token F1 (%)")
ax.set_xticks(x)
ax.set_xticklabels(types)
ax.set_ylim(0, 35)
ax.legend()
ax.grid(axis="y", alpha=0.25)

ax.bar_label(bars1, fmt="%.2f", padding=3)
ax.bar_label(bars2, fmt="%.2f", padding=3)

save_figure(fig, "02_question_type_comparison")


# 3. 效率与性能关系
fig, ax = plt.subplots(figsize=(8.5, 5.5))

ax.scatter(runtime, f1, s=100)

for name, t, score in zip(methods, runtime, f1):
    ax.annotate(name, (t, score),
                xytext=(7, 7),
                textcoords="offset points")

ax.set_title("Runtime vs Answer Quality")
ax.set_xlabel("Average Runtime (seconds)")
ax.set_ylabel("Token F1 (%)")
ax.grid(alpha=0.25)

save_figure(fig, "03_runtime_vs_f1")


# 4. 成功率对比
fig, ax = plt.subplots(figsize=(8.5, 5.5))
bars = ax.bar(methods, success_rate)

ax.set_title("Execution Success Rate")
ax.set_ylabel("Success Rate (%)")
ax.set_ylim(85, 100)
ax.grid(axis="y", alpha=0.25)
ax.bar_label(bars, fmt="%.2f", padding=3)

save_figure(fig, "04_success_rate")


print("\n全部可视化图表生成完成！")
print(f"输出目录：{OUTPUT_DIR.resolve()}")

