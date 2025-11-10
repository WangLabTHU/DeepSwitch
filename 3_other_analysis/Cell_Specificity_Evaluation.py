import torch.nn as nn
import torch
import pandas as pd
import numpy as np

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
        # x = torch.sigmoid(x)

        return x


if __name__ == '__main__':
    # Load data
    data = pd.read_csv('DeepCP_dataset/CP_discriminator.csv')
    filtered_data_E1 = data[data['Label'] == 0]['Sequences'].tolist()
    data = pd.read_csv('DeepCP_dataset/switch_or_not.csv')
    filtered_data_E2 = data[data['Label'] == 0]['Sequences'].tolist()
    data = pd.read_csv('DeepCP_dataset/switch_60bp.csv')
    filtered_data_E3 = data['Sequences'].tolist()
    sequences = filtered_data_E1 + filtered_data_E2 + filtered_data_E3

    seq = torch.tensor(sequences_to_one_hot(sequences), dtype=torch.float32)
    batch_size = 40960

    # (-Dox)
    model = CNNModel(seq_len=60, kernel_sizes=5)
    save_path = 'Model/Natural_AI_generated_(-Dox)_model.pth'
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    pred_dox_minus_list = []
    with torch.no_grad():
        for i in range(0, len(seq), batch_size):
            batch_seq = seq[i:i + batch_size].to(device)
            outputs = model(batch_seq)
            pred_yes_no = outputs.squeeze().detach().cpu().numpy().tolist()
            pred_dox_minus_list.extend(pred_yes_no)

    # (+Dox)
    model = CNNModel(seq_len=60, kernel_sizes=5)
    save_path = 'Model/Natural_AI_generated_(+Dox)_model.pth'
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    pred_dox_plus_list = []
    with torch.no_grad():
        for i in range(0, len(seq), batch_size):
            batch_seq = seq[i:i + batch_size].to(device)
            outputs = model(batch_seq)
            pred_yes_no = outputs.squeeze().detach().cpu().numpy().tolist()
            pred_dox_plus_list.extend(pred_yes_no)

    custom_label = [0] * len(filtered_data_E1) + [1] * len(filtered_data_E2) + [2] * len(filtered_data_E3)
    df = pd.DataFrame({'Sequences': sequences, 'type': custom_label, 'Pred_dox_minus': pred_dox_minus_list, 'Pred_dox_plus': pred_dox_plus_list})
    df.to_csv('DeepCP_dataset/Cell_specificity_evaluation_prediction.csv', index=False)
