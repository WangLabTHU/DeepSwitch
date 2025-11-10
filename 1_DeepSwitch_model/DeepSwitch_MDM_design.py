import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import pickle

# GPU or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Sequence to one-hot encoding
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


def model_predict(sequences):
    seq = torch.tensor(sequences_to_one_hot(sequences), dtype=torch.float32)
    batch_size = 40960

    # Switch Discriminator
    model = CNNModel(seq_len=60, kernel_sizes=5)
    model.to(device)
    model_state = torch.load('Model/best_discriminator.pth')
    model.load_state_dict(model_state)
    model.eval()

    pred_yes_no_list = []
    with torch.no_grad():
        for i in range(0, len(seq), batch_size):
            batch_seq = seq[i:i + batch_size].to(device)
            outputs = model(batch_seq)
            pred_yes_no = outputs.squeeze().detach().cpu().numpy().tolist()
            pred_yes_no_list.extend(pred_yes_no)

    # Evaluator
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

    # Switch identification
    probability_final = [[], []]
    for i in range(len(pred_yes_no_list)):
        if pred_yes_no_list[i] < 0.5:
            probability_final[0].append(0)
            probability_final[1].append(0)
        else:
            probability_final[0].append(Dox_minus_score_list[i])
            probability_final[1].append(Dox_plus_score_list[i])

    # Monitor
    model = CNNModel(seq_len=60, kernel_sizes=5)
    model.to(device)
    model_state = torch.load('Model/best_monitor.pth')
    model.load_state_dict(model_state)
    model.eval()

    pred_out_of_dis = []
    with torch.no_grad():
        for i in range(0, len(seq), batch_size):
            batch_seq = seq[i:i + batch_size].to(device)
            outputs = model(batch_seq)
            out_of_dis_score = outputs.squeeze().detach().cpu().numpy().tolist()
            pred_out_of_dis.extend(out_of_dis_score)

    #   CP identification
    for i in range(len(pred_out_of_dis)):
        if pred_out_of_dis[i] < 0.5:
            probability_final[0][i] = 0
            probability_final[1][i] = 0

    return probability_final


def Predict_results(Sequences):
    Score = model_predict(Sequences)
    print(len(Score), len(Score[0]))

    filename = 'Updated_DeepCP/score_data.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(Score, f)
    print(f"Score data has been saved to {filename}")


def filter_sequences_scores(sequences, score):
    # 创建新的列表来存储筛选后的序列和分数
    filtered_sequences = []
    filtered_scores = []

    # 遍历所有序列和分数
    for seq, s0, s1 in zip(sequences, score[0], score[1]):
        if s0 != 0 and s1 != 0:
            filtered_sequences.append(seq)
            filtered_scores.append([s0, s1])

    return filtered_sequences, filtered_scores


def select_top_sequences(filtered_sequences, filtered_scores):
    # 转换为 numpy 数组以便操作
    sequences = np.array(filtered_sequences)
    scores = np.array(filtered_scores)

    # 第一步：按filtered_scores[1]降序排序，取前25000个
    # top_250k_indices = np.argsort(scores[:, 1])[::-1][:1000000]  # 按score2降序排列
    # top_250k_sequences = sequences[top_250k_indices]
    # top_250k_scores = scores[top_250k_indices]
    # 32843个
    # score2_threshold = 1.8934459740599998
    # top_indices = np.where(scores[:, 1] >= score2_threshold)[0]
    # top_indices_sorted = top_indices[np.argsort(scores[top_indices, 1])[::-1]]
    # top_sequences = sequences[top_indices_sorted]
    # top_scores = scores[top_indices_sorted]

    score1_threshold = 1.02360615692
    score2_threshold = 1.8934459740599998
    top_indices = np.where((scores[:, 0] <= score1_threshold) & (scores[:, 1] >= score2_threshold))[0]
    top_indices_sorted = top_indices[np.argsort(scores[top_indices, 1])[::-1]]
    top_sequences = sequences[top_indices_sorted]
    top_scores = scores[top_indices_sorted]
    print(len(top_scores))

    # 第二步：在这250000个点中，按filtered_scores[0]升序排序，取前2500个
    ratios = top_scores[:, 1] / top_scores[:, 0]
    top_2500_indices = np.argsort(ratios)[::-1][:2500]
    top_2500_sequences = top_sequences[top_2500_indices]
    top_2500_scores = top_scores[top_2500_indices]

    return top_2500_sequences, top_2500_scores


def Find_top(Sequences):
    with open('Updated_DeepCP/score_data.pkl', 'rb') as f:
        Score = pickle.load(f)
    # print(len(Score), len(Score[0]), max(Score[1]))
    filtered_sequences, filtered_scores = filter_sequences_scores(Sequences, Score)
    top_2500_sequences, top_2500_scores = select_top_sequences(filtered_sequences, filtered_scores)

    res = pd.DataFrame({'top_sequences': top_2500_sequences,
                        'Dox_minus': [score[0] for score in top_2500_scores],
                        'Dox_plus': [score[1] for score in top_2500_scores]})
    res.to_csv('Updated_DeepCP/Updated_DeepCP_Diffusion_sequences_r1.csv')


if __name__ == '__main__':
    # Dataset Load
    Sequences = []
    with open('MPRA_sequences/Diffusion_0_5000000.txt', 'r') as myfile:
        for line in myfile:
            if line[0] != '>':
                Sequences.append(line.strip('\n'))
    # Predict_results(Sequences)
    Find_top(Sequences)
