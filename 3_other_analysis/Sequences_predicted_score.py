import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random

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


def model_predict(sequences):
    seq = torch.tensor(sequences_to_one_hot(sequences), dtype=torch.float32)
    batch_size = 40960

    # Evaluator
    model = CNNModel(seq_len=60, kernel_sizes=5)
    model.to(device)
    model_state = torch.load('Model/best_evaluator_specificity.pth')
    model.load_state_dict(model_state)
    model.eval()

    specificity_score_list = []
    with torch.no_grad():
        for i in range(0, len(seq), batch_size):
            batch_seq = seq[i:i + batch_size].to(device)
            outputs = model(batch_seq)
            specificity_score = outputs.squeeze().detach().cpu().numpy().tolist()
            specificity_score_list.extend(specificity_score)

            del batch_seq
            del outputs
            torch.cuda.empty_cache()

    model = CNNModel(seq_len=60, kernel_sizes=5)
    model.to(device)
    model_state = torch.load('Model/best_evaluator_strength.pth')
    model.load_state_dict(model_state)
    model.eval()

    strength_score_list = []
    with torch.no_grad():
        for i in range(0, len(seq), batch_size):
            batch_seq = seq[i:i + batch_size].to(device)
            outputs = model(batch_seq)
            strength_score = outputs.squeeze().detach().cpu().numpy().tolist()
            strength_score_list.extend(strength_score)

            del batch_seq
            del outputs
            torch.cuda.empty_cache()

    return specificity_score_list, strength_score_list


if __name__ == '__main__':
    # Dataset Load
    df = pd.read_csv('MPRA_sequences/CP_Exp.csv')
    Sequences = df['DNA Sequence'].to_list()

    specificity_score_list, strength_score_list = model_predict(Sequences)
    res = pd.DataFrame({'Sequences': Sequences,
                        'specificity_score': specificity_score_list,
                        'strength_score_list': strength_score_list})

    res.to_csv('MPRA_sequences/CP_predicted_score.csv')
