import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
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


# Label processing
def process_labels(labels):
    return np.array(labels)


# Dataset generation
def Generate_input(pre_data):
    # dataset preprocessing
    sequences = pre_data['DNA Sequence'].tolist()
    labels_a = pre_data['(-Dox)'].tolist()
    labels_b = pre_data['(+Dox)'].tolist()

    sequences = sequences_to_one_hot(sequences)
    labels_a = process_labels(labels_a)
    labels_b = process_labels(labels_b)

    # Tensor
    dataset = TensorDataset(torch.tensor(sequences, dtype=torch.float32),
                            torch.tensor(labels_a, dtype=torch.float32),
                            torch.tensor(labels_b, dtype=torch.float32))
    return dataset


# Dataset Division
def prepare_data():
    df = pd.read_csv('DeepCP_dataset/CP_Exp.csv')
    # df['(-Dox)'] = np.log10(df['(-Dox)'])
    # df['(+Dox)'] = np.log10(df['(+Dox)'])
    # min_value = df['(-Dox)'].min()
    # max_value = df['(-Dox)'].max()
    # df['(-Dox)'] = (df['(-Dox)'] - min_value) / (max_value - min_value) + 0.1
    # min_value = df['(+Dox)'].min()
    # max_value = df['(+Dox)'].max()
    # df['(+Dox)'] = (df['(+Dox)'] - min_value) / (max_value - min_value) + 0.1
    # data = df[df['Identifier'].str.contains('Natural_1|Natural_0|Diffusion|GA')]
    data = df
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # Generate dataset
    train_dataset = Generate_input(train_data)
    val_dataset = Generate_input(val_data)
    test_dataset = Generate_input(test_data)

    return train_dataset, val_dataset, test_dataset, train_data, val_data, test_data


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


# Model evaluation
def evaluate_model(model, criterion, data_loader):
    model.eval()

    loss = 0
    predicted_labels_a = []
    true_labels_a = []
    predicted_labels_b = []
    true_labels_b = []

    with torch.no_grad():
        for inputs, targets_a, targets_b in data_loader:
            inputs = inputs.to(device)
            targets_a = targets_a.to(device).view(-1, 1)  # 调整 targets_a 形状
            targets_b = targets_b.to(device).view(-1, 1)  # 调整 targets_b 形状

            outputs_a, outputs_b = model(inputs)

            # 计算损失
            loss_a = criterion(outputs_a, targets_a)
            loss_b = criterion(outputs_b, targets_b)
            loss += (loss_a + loss_b).item() / 2

            # 保存预测结果和真实值
            predicted_labels_a.extend(outputs_a.cpu().numpy())
            true_labels_a.extend(targets_a.cpu().numpy())
            predicted_labels_b.extend(outputs_b.cpu().numpy())
            true_labels_b.extend(targets_b.cpu().numpy())

    # 计算平均损失
    average_loss = loss / len(data_loader)
    pcc_a, _ = pearsonr(np.squeeze(predicted_labels_a), np.squeeze(true_labels_a))
    rmse_a = np.sqrt(np.mean((np.array(predicted_labels_a) - np.array(true_labels_a)) ** 2))
    pcc_b, _ = pearsonr(np.squeeze(predicted_labels_b), np.squeeze(true_labels_b))
    rmse_b = np.sqrt(np.mean((np.array(predicted_labels_b) - np.array(true_labels_b)) ** 2))

    return average_loss, pcc_a, rmse_a, pcc_b, rmse_b


def save_predictions(model, data_loaders, datasets, output_file='Updated_DeepCP/Natural_AI_predictions.csv'):
    model.eval()
    all_predictions_a = []
    all_predictions_b = []
    all_sequences = []
    all_true_labels_a = []
    all_true_labels_b = []
    all_set_labels = []

    with torch.no_grad():
        for set_label, (data_loader, dataset) in enumerate(zip(data_loaders, datasets)):
            for i, (inputs, targets_a, targets_b) in enumerate(data_loader):
                inputs = inputs.to(device)
                outputs_a, outputs_b = model(inputs)
                outputs_a = outputs_a.detach().cpu().numpy().squeeze()
                outputs_b = outputs_b.detach().cpu().numpy().squeeze()

                all_predictions_a.extend(outputs_a)
                all_predictions_b.extend(outputs_b)
                all_true_labels_a.extend(targets_a.numpy())
                all_true_labels_b.extend(targets_b.numpy())
                all_sequences.extend([dataset.iloc[j]['DNA Sequence'] for j in range(i * data_loader.batch_size, min((i + 1) * data_loader.batch_size, len(dataset)))])
                all_set_labels.extend([set_label] * inputs.shape[0])

    # 保存到CSV
    df = pd.DataFrame({
        'Sequence': all_sequences,
        'True Label (-Dox)': all_true_labels_a,
        'Predicted Label (-Dox)': all_predictions_a,
        'True Label (+Dox)': all_true_labels_b,
        'Predicted Label (+Dox)': all_predictions_b,
        'Set': all_set_labels
    })

    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


def huber_loss(y_pred, y_true, delta=2.0):
    error = y_pred - y_true
    squared_loss = 0.5 * error ** 2  # 误差小时用MSE
    absolute_loss = delta * (torch.abs(error) - 0.5 * delta)  # 误差大时用L1
    return torch.where(torch.abs(error) <= delta, squared_loss, absolute_loss).mean()


def main():
    # Dataset load
    train_dataset, val_dataset, test_dataset, train_data, val_data, test_data = prepare_data()
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Model initialization
    model = TwinTowerCNNModel(seq_len=60, kernel_sizes=5)
    model.to(device)

    # Parameters definition
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    best_val_loss = float('inf')
    best_model_state_dict = None

    for epoch in range(num_epochs):
        model.train()
        # Model training
        for inputs, targets_a, targets_b in train_loader:
            inputs = inputs.to(device)
            targets_a = targets_a.to(device).view(-1, 1).to(device)  # 调整 targets_a 形状
            targets_b = targets_b.to(device).view(-1, 1).to(device)  # 调整 targets_b 形状

            optimizer.zero_grad()
            outputs_a, outputs_b = model(inputs)

            # 计算损失
            loss_a = criterion(outputs_a, targets_a)
            loss_b = criterion(outputs_b, targets_b)
            loss = (loss_a + loss_b)/2

            loss.backward()
            optimizer.step()

        # Model validation
        average_loss, pcc_a, rmse_a, pcc_b, rmse_b = evaluate_model(model, criterion, val_loader)
        print(f'Validation PCC (-Dox): {pcc_a}, RMSE (-Dox): {rmse_a}')
        print(f'Validation PCC (+Dox): {pcc_b}, RMSE (+Dox): {rmse_b}')

        if average_loss < best_val_loss:
            best_val_loss = average_loss
            best_model_state_dict = model.state_dict()

    # Model save
    # save_path = 'Updated_DeepCP/Natural_AI_two_tower_model.pth'
    # torch.save(best_model_state_dict, save_path)

    # Load the best model
    model.load_state_dict(best_model_state_dict)

    # Final model evaluation on test set
    average_loss, pcc_a, rmse_a, pcc_b, rmse_b = evaluate_model(model, criterion, test_loader)
    print('*****************')
    print('Test Set Evaluation')
    print(f'Test PCC (-Dox): {pcc_a}, RMSE (-Dox): {rmse_a}')
    print(f'Test PCC (+Dox): {pcc_b}, RMSE (+Dox): {rmse_b}')

    # Save predictions
    save_predictions(model, [train_loader, val_loader, test_loader], [train_data, val_data, test_data],
                     'Updated_DeepCP/Natural_AI_predictions_r.csv')


if __name__ == "__main__":
    main()
