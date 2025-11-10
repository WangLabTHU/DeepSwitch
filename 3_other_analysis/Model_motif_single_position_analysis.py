import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################### model loading
def sequences_to_one_hot(sequences):
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    one_hot_sequences = [np.zeros((len(seq), len(mapping.keys()))) for seq in sequences]

    for i, seq in enumerate(sequences):
        for j, nucleotide in enumerate(seq):
            one_hot_sequences[i][j, mapping[nucleotide]] = 1

    return np.array(one_hot_sequences)


class CNNModel(nn.Module):
    def __init__(self, seq_len=100, kernel_sizes=5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=(kernel_sizes,))
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=(kernel_sizes,))
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=(kernel_sizes,))
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(32, 16, 1, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(int((seq_len-kernel_sizes*3+3)*0.5)*32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # Convolution
        x = x.permute(0, 2, 1)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)

        # lstm
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        # dense
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


class TwinTowerCNNModel(nn.Module):
    def __init__(self, seq_len=100, kernel_sizes=5):
        super(TwinTowerCNNModel, self).__init__()
        # Shared layers
        self.conv_shared1 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=kernel_sizes)
        self.bn_shared1 = nn.BatchNorm1d(256)
        self.conv_shared2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=kernel_sizes)
        self.bn_shared2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)

        # Tower (-Dox)
        self.conv_m1_a = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=kernel_sizes)
        self.bn_m1_a = nn.BatchNorm1d(64)
        self.conv_m2_a = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=kernel_sizes)
        self.bn_m2_a = nn.BatchNorm1d(32)
        self.conv_m3_a = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=4)

        # Tower (+Dox)
        self.conv_m1_b = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=kernel_sizes)
        self.bn_m1_b = nn.BatchNorm1d(64)
        self.conv_m2_b = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=kernel_sizes)
        self.bn_m2_b = nn.BatchNorm1d(32)
        self.conv_m3_b = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=4)

        # Other layers
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        # Shared layers
        x = self.conv_shared1(x)
        x = self.bn_shared1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv_shared2(x)
        x = self.bn_shared2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Tower (-Dox)
        x_a = self.conv_m1_a(x)
        x_a = self.bn_m1_a(x_a)
        x_a = self.relu(x_a)
        x_a = self.conv_m2_a(x_a)
        x_a = self.bn_m2_a(x_a)
        x_a = self.relu(x_a)
        x_a = self.conv_m3_a(x_a)
        x_a = self.flatten(x_a)
        x_a = self.dropout(x_a)

        # Tower (+Dox)
        x_b = self.conv_m1_b(x)
        x_b = self.bn_m1_b(x_b)
        x_b = self.relu(x_b)
        x_b = self.conv_m2_b(x_b)
        x_b = self.bn_m2_b(x_b)
        x_b = self.relu(x_b)
        x_b = self.conv_m3_b(x_b)
        x_b = self.flatten(x_b)
        x_b = self.dropout(x_b)

        return x_a, x_b


####################### motif analysis
def model_predict(sequences):
    seq = torch.tensor(sequences_to_one_hot(sequences), dtype=torch.float32)
    batch_size = 40960
    model = TwinTowerCNNModel(seq_len=60, kernel_sizes=5)
    model.to(device)
    model_state = torch.load('Updated_DeepCP/Natural_AI_two_tower_model.pth')
    model.load_state_dict(model_state)
    model.eval()

    Dox_minus_score_list = []
    Dox_plus_score_list = []
    with torch.no_grad():
        for i in range(0, len(seq), batch_size):
            batch_seq = seq[i:i + batch_size].to(device)
            outputs_a, outputs_b = model(batch_seq)

            dox_minus_score = outputs_a.squeeze().detach().cpu().numpy().tolist()
            Dox_minus_score_list.extend(dox_minus_score)

            dox_plus_score = outputs_b.squeeze().detach().cpu().numpy().tolist()
            Dox_plus_score_list.extend(dox_plus_score)

    Dox_minus_score = np.array(Dox_minus_score_list)
    Dox_plus_score = np.array(Dox_plus_score_list)
    return Dox_minus_score, Dox_plus_score


def generate_background_sequences(n, seq_len=60):
    random.seed(99)
    df = pd.read_csv("/home/hyu/DeepCP/DeepCP_dataset/CP_discriminator.csv")
    seqs = df[df['Label']==1]['Sequences'].tolist()
    return random.sample(seqs, n)

def replace_motif_at_pos(seqs, motif, pos):
    return [s[:pos] + motif + s[pos+len(motif):] for s in seqs]

def analyze_motif(motif, conserved_pos, n_per_pos=1000, seq_len=60):
    pos0 = conserved_pos - 1
    bg = generate_background_sequences(n_per_pos, seq_len)
    mod = replace_motif_at_pos(bg, motif, pos0)
    b_minus, b_plus = model_predict(bg)
    m_minus, m_plus = model_predict(mod)
    relative_fc = (m_plus/m_minus) / (b_plus/b_minus)
    relative_plus = m_plus/(b_plus)
    relative_minus = m_minus/(b_minus)
    return relative_minus, relative_plus, relative_fc


if __name__ == '__main__':
    motifs_info = {
        "KLF5": (1, "GCCCCGCCCC"),
        "Plagl1": (1, "CCCTGGGGCCAGG"),
        "ZNF418": (15, "AAGAGGCTAAAAGCA"),
        "TFAP2A": (20, "TGCCCTGAGGGCA"),
        "ASCL1": (29, "CAGCACCTGCCCC"),
        "Ptf1A": (29, "GCACAGCTGTGC"),
        "TATA-box": (20, "TATAAAA"),
        "Initiator": (48, "GCCAGT"),
        "BREu": (13, "GCGCGCC"),
        "BREd": (28, "GTTTGTT")
    }

    save_dir = "Figures"
    os.makedirs(save_dir, exist_ok=True)
    labels, data = [], []
    for name, (pos, seq) in motifs_info.items():
        print(f"Processing {name} ...")
        minus, plus, fc = analyze_motif(seq, pos)
        data.extend(minus)
        data.extend(plus)
        data.extend(fc)

    for Type in motifs_info.keys(): 
        for cat in ['_basal', '_induced', '_fold change']: 
            labels.extend([Type + cat] * 1000)
    
    # print(len(data), len(data[0]))
    # print(len(labels), labels[0])
    df_plot = pd.DataFrame({'Value': data, 'Category': labels})
    categories = df_plot['Category'].unique()
    p_values = []
    mean_values = []
    for cat in categories:
        values = df_plot.loc[df_plot['Category'] == cat, 'Value']
        stat, p = wilcoxon(values - 1, zero_method='wilcox', alternative='two-sided')
        p_values.append(p)
        mean_values.append(values.median())
    # print(p_values)
    significance = []
    for p in p_values:
        if p < 0.001:
            significance.append('***')
        elif p < 0.01:
            significance.append('**')
        elif p < 0.05:
            significance.append('*')
        else:
            significance.append('NS')

    lower = df_plot['Value'].quantile(0.025)
    upper = df_plot['Value'].quantile(0.975)
    df_plot = df_plot[(df_plot['Value'] >= lower) & (df_plot['Value'] <= upper)]

    # 绘制 boxplot
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    palette_dict = {cat: colors[i % 3] for i, cat in enumerate(categories)}
    ax = sns.boxplot(x='Category', y='Value', hue='Category', data=df_plot, medianprops={"color": "k", "linewidth": 1},
                     showfliers=False, linewidth=1.8, width=0.6, palette=palette_dict)
    hline = plt.axhline(y=1, color='k', linestyle='--', linewidth=1.5, label="y=1")
    plt.xlabel('')
    plt.ylabel('Relative Ratio', fontsize=24)
    plt.xticks([], fontsize=20)
    plt.yticks([], fontsize=20)

    motif_names = motifs_info.keys()
    ax.set_xticks([1, 4, 7, 10, 13, 16, 19, 22, 25, 28])
    ax.set_xticklabels(motif_names, fontsize=20, rotation=0, ha='center')
    patches = [mpatches.Patch(color=colors[i], label=lab) for i, lab in enumerate(['Predicted -Dox', 'Predicted +Dox', 'Predicted Fold change'])]
    handles = patches + [hline]
    for i in range(3, 30, 3):
        ax.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1)
    ymin, ymax = ax.get_ylim()
    offset = 0.02
    for i, sig in enumerate(significance):
        ax.text(i, ymax + offset, sig, ha='center', va='bottom',
                fontsize=16, color='k', fontfamily='monospace')
    plt.legend(handles=handles, fontsize=16, loc='upper right')

    # --- 标注显著性和箭头 ---
    for i, (p, mean_val) in enumerate(zip(p_values, mean_values)):
        if p < 0.05:
            if mean_val > 1:
                ax.text(i, ymax + offset + 0.15, '↑', ha='center', va='bottom', fontsize=20, color='k')
            else:
                ax.text(i, ymax + offset + 0.15, '↓', ha='center', va='bottom', fontsize=20, color='k')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'All_motifs_boxplot.png'), dpi=400)
