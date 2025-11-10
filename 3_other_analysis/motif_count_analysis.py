import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os

def plot_motif_snvs_distribution_pie(seqs, motif="TATAAAA", save_path="Figures/motif_mutation_distribution.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bases = ['A', 'T', 'C', 'G']
    counts = Counter()
    for seq in seqs:
        if motif in seq:
            counts[motif] += 1
        for i in range(2, len(motif)):
            for b in bases:
                if b != motif[i]:
                    variant = motif[:i] + b + motif[i+1:]
                    if variant in seq:
                        label = f"{variant} (pos{i+1}:{motif[i]}->{b})"
                        counts[label] += 1
    # total_counts = sum(counts.values())
    # print(f"Total counts (all motif variants): {total_counts}")
    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    colors = plt.get_cmap('Pastel1').colors
    color_list = [colors[i % len(colors)] for i in range(len(counts))]
    plt.figure(figsize=(10, 10))
    plt.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%", startangle=170, colors=color_list, textprops={'fontsize': 12})
    plt.tight_layout()
    # plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Pie chart saved to {save_path}")

df = pd.read_csv("/home/hyu/DeepCP/DeepCP_dataset/CP_discriminator.csv")
seqs = df[df['Label'] == 1]['Sequences'].tolist()

plot_motif_snvs_distribution_pie(seqs)
