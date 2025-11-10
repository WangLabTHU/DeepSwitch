import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os

# 保存目录
save_dir = "/home/hyu/DeepCP/Figures"
os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建

# 读取数据
df = pd.read_csv("/home/hyu/DeepCP/DeepCP_dataset/CP_Exp.csv")
sequences = df["DNA Sequence"].tolist()
neg_dox = df["(-Dox)"].tolist()
pos_dox = df["(+Dox)"].tolist()
# df = pd.read_csv("/home/hyu/DeepCP/DeepCP_dataset/MPRA_1st_library.tsv", sep='\t', header=0)
# sequences = df["sequence"].tolist()
# neg_dox = df["HEK293T"].tolist()
# pos_dox = df["HepG2"].tolist()

# 定义 motifs
motifs = {
    'KLF5': 'GCCCCGCCCC', 'Initiator': 'GCCAGT',
    'BREu': 'GCGCGCC', 'BREd': 'GTTTGTT', 'TATA box': 'TATAAAA'
}

# 保存结果: motif -> position -> list of values
results = {name: defaultdict(lambda: {"neg": [], "pos": [], "FC": []})
           for name in motifs}

# 遍历序列并记录 motif 出现位置及指标值
for seq, nd, pdx in zip(sequences, neg_dox, pos_dox):
    ratio = pdx / nd if nd != 0 else float('nan')  # 避免除以0
    for name, motif in motifs.items():
        for m in re.finditer(motif, seq):
            pos = m.start()
            results[name][pos]["neg"].append(nd)
            results[name][pos]["pos"].append(pdx)
            results[name][pos]["FC"].append(ratio)

# 绘图并保存
for name in motifs:
    # 统计每个位置的 motif 出现频率
    pos_counts = {pos: len(results[name][pos]["neg"]) for pos in results[name]}
    
    # 先过滤掉样本数 < 10 的位置
    filtered_positions = {pos: cnt for pos, cnt in pos_counts.items() if cnt >= 10}
    
    # 在过滤后的结果里选取 top10
    top_positions = [p for p, _ in Counter(filtered_positions).most_common(10)]
    top_positions = sorted(top_positions)  # 排序，便于画图

    # 如果没有符合条件的位置，跳过
    if not top_positions:
        continue

    # 构建绘图数据
    plot_data = []
    for pos in top_positions:
        for val in results[name][pos]["pos"]:
            plot_data.append({"Position": pos, "Type": "+Dox", "Value": val})
        for val in results[name][pos]["FC"]:
            plot_data.append({"Position": pos, "Type": "FC", "Value": val})
    plot_df = pd.DataFrame(plot_data)

    # 绘制violin plot（固定宽度 & 美化）
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=plot_df, 
        x="Position", 
        y="Value", 
        hue="Type",
        split=True, 
        inner="quartile",   # 展示四分位线
        cut=0, 
        linewidth=1.2,
        palette="Set2",     # 柔和配色
        width=0.8,
        density_norm='width'
    )

    plt.title(f"Motif: {name} (n>=10)", fontsize=14, weight="bold")
    plt.xlabel("Motif start position", fontsize=12)
    plt.ylabel("Value distribution", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="Category", fontsize=10, title_fontsize=11, loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # 保存图像（保持你原来的命名风格）
    save_path = os.path.join(save_dir, f"{name}_motif_top10_distribution.png")
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
