import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from imblearn.under_sampling import RandomUnderSampler


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
    sequences = pre_data['Sequences'].tolist()
    labels = pre_data['Label'].tolist()
    sequences = sequences_to_one_hot(sequences)
    labels = process_labels(labels)

    # dataset balance
    seq_length = sequences.shape[1]
    seq_map = sequences.shape[2]
    sequences_numeric = sequences.reshape([-1, seq_length*seq_map])
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    sequences_numeric, labels = undersampler.fit_resample(sequences_numeric, labels)
    sequences_numeric = sequences_numeric.reshape([-1, seq_length, seq_map])

    # Tensor
    dataset = TensorDataset(torch.tensor(sequences_numeric, dtype=torch.float32),
                            torch.tensor(labels, dtype=torch.float32))
    return dataset


# Dataset Division
def prepare_data():
    data = pd.read_csv('DeepCP_dataset/switch_or_not.csv')

    # Chromosome Division
    chr19_data = data[data['Chromosome'] == 'chr19']
    chr21_data = data[data['Chromosome'] == 'chr21']
    chrX_data = data[data['Chromosome'] == 'chrX']
    chr7_data = data[data['Chromosome'] == 'chr7']
    chr13_data = data[data['Chromosome'] == 'chr13']

    # Dataset Division by chromosome
    val_data = pd.concat([chr19_data, chr21_data, chrX_data])
    test_data = pd.concat([chr7_data, chr13_data])
    remaining_data = data.drop(val_data.index)
    train_data = remaining_data.drop(test_data.index)

    # Generate dataset
    train_dataset = Generate_input(train_data)
    val_dataset = Generate_input(val_data)
    test_dataset = Generate_input(test_data)

    return train_dataset, val_dataset, test_dataset


# Model class
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


# Model evaluation
def evaluate_model(model, criterion, data_loader, save_auc_data_path=None):
    model.eval()

    # Calculate loss and append probabilities and true labels
    loss = 0
    predicted_probs = []
    true_labels = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Calculate loss
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.detach().cpu().squeeze()
            loss += criterion(outputs, targets).item()

            # Append the probabilities and true labels
            predicted_probs.extend(outputs)
            true_labels.extend(targets.numpy())

    # Calculate evaluation metrics
    average_loss = loss / len(data_loader)

    y_true = np.array(true_labels)
    y_score = np.array(predicted_probs)
    if save_auc_data_path is not None:
        np.savez(save_auc_data_path, y_true=y_true, y_score=y_score)

    accuracy = accuracy_score(true_labels, (np.array(predicted_probs) > 0.5).astype(int))
    precision = precision_score(true_labels, (np.array(predicted_probs) > 0.5).astype(int))
    recall = recall_score(true_labels, (np.array(predicted_probs) > 0.5).astype(int))
    f1 = f1_score(true_labels, (np.array(predicted_probs) > 0.5).astype(int))
    auc_roc = roc_auc_score(true_labels, predicted_probs)
    auc_pr = average_precision_score(true_labels, predicted_probs)
    return average_loss, accuracy, precision, recall, f1, auc_roc, auc_pr


def main():
    # Dataset load
    train_dataset, val_dataset, test_dataset = prepare_data()
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Model initialization
    model = CNNModel(seq_len=60, kernel_sizes=5)
    model.to(device)

    # Parameters definition
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    best_val_loss = float('inf')
    best_model_state_dict = None

    for epoch in range(num_epochs):
        model.train()
        # Model training
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()

        # Model validation
        average_loss, accuracy, precision, recall, f1, auc_roc, _ = evaluate_model(model, criterion, val_loader)
        print('Epoch ', epoch+1)
        print(f'Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}')
        if average_loss < best_val_loss:
            best_val_loss = average_loss
            best_model_state_dict = model.state_dict()

    # Model save
    # save_path = 'Model/best_discriminator.pth'
    # torch.save(best_model_state_dict, save_path)

    # Model evaluation
    model.load_state_dict(best_model_state_dict)
    average_loss, accuracy, precision, recall, f1, auc_roc, auc_pr = evaluate_model(model, criterion, test_loader, save_auc_data_path="Figures/Discriminator_roc.npz")
    print('Model Evaluation')
    print(f'Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}')


if __name__ == "__main__":
    main()
