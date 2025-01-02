import os
import argparse
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from scripts.dataset import KeypointsSequenceDataset
from scripts.model import GAT_LSTM_Attn_Model
from scripts.util import save_model, save_confusion_matrix

def train_client(model, train_loader, criterion, optimizer, device, num_epochs=3, global_model=None, method="FedAvg", mu=0.01):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            edge_index = torch.tensor([
                [0, 1], [0, 2], [1, 3], [2, 4],
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                [5, 11], [6, 12], [11, 12], 
                [11, 13], [12, 14], [13, 15], [14, 16]
            ], dtype=torch.long).t().contiguous().to(device)

            outputs, _, _, _, _ = model(sequences, edge_index)
            loss = criterion(outputs, labels)

            if method == "FedProx" and global_model is not None:
                prox_term = sum(torch.sum((param - global_param) ** 2)
                                for param, global_param in zip(model.parameters(), global_model.parameters()))
                loss += (mu / 2) * prox_term

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

    return epoch_loss, epoch_acc

def federated_train(global_model, train_loaders, val_loaders, test_loader, num_rounds, log_file, method="FedAvg", mu=0.01):
    with open(log_file, 'w') as f:
        f.write('Round,Train Loss,Train Acc,Val Loss,Val Acc,Val Precision,Val Recall,Val F1\n')

    for round in range(num_rounds):
        local_weights = []
        local_steps = []

        for train_loader in train_loaders:
            local_model = GAT_LSTM_Attn_Model(sequence_length=40, in_channels=3, hidden_channels=64, out_channels=128, num_classes=2).to(global_model.device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.Adam(local_model.parameters(), lr=0.001)

            train_loss, train_acc = train_client(local_model, train_loader, criterion, optimizer, global_model.device,
                                                 num_epochs=3, global_model=global_model, method=method, mu=mu)

            local_weights.append(local_model.state_dict())
            local_steps.append(len(train_loader))

        global_state_dict = global_model.state_dict()

        if method == "FedAvg" or method == "FedProx":
            for key in global_state_dict.keys():
                global_state_dict[key] = torch.mean(torch.stack([local_weight[key] for local_weight in local_weights]), dim=0)

        elif method == "FedNova":
            total_steps = sum(local_steps)
            normalized_weights = [steps / total_steps for steps in local_steps]

            for key in global_state_dict.keys():
                aggregated_value = torch.zeros_like(global_state_dict[key])
                for local_weight, weight in zip(local_weights, normalized_weights):
                    aggregated_value += weight * (local_weight[key] - global_state_dict[key])
                global_state_dict[key] += aggregated_value

        global_model.load_state_dict(global_state_dict)

        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(global_model, val_loaders[0], criterion, global_model.device)
        print(f'Round {round+1}/{num_rounds}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

        with open(log_file, 'a') as f:
            f.write(f'{round+1},{val_loss:.4f},{val_acc:.4f},{val_precision:.4f},{val_recall:.4f},{val_f1:.4f}\n')

    test_loss, test_acc, test_precision, test_recall, test_f1, test_labels, test_predictions = evaluate(global_model, test_loader, criterion, global_model.device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    print(f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')

    conf_matrix = confusion_matrix(test_labels, test_predictions)
    save_confusion_matrix(conf_matrix, list(label_map.keys()), "./outputs/confusion_matrix.png")
    save_model(global_model, "./outputs/federated_model.pth")

def main():
    parser = argparse.ArgumentParser(description="Federated Training for GAT-LSTM")
    parser.add_argument("--method", type=str, choices=["FedAvg", "FedProx", "FedNova"], default="FedAvg", help="Federated learning method")
    parser.add_argument("--mu", type=float, default=0.01, help="Proximal term coefficient for FedProx")
    parser.add_argument("--num_rounds", type=int, default=100, help="Number of federated learning rounds")
    parser.add_argument("--log_file", type=str, default="./outputs/federated_training_log.txt", help="Path to the log file")
    args = parser.parse_args()

    global_model = GAT_LSTM_Attn_Model(sequence_length=40, in_channels=3, hidden_channels=64, out_channels=128, num_classes=2).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    federated_train(
        global_model=global_model,
        train_loaders=[train_loader_1, train_loader_2],
        val_loaders=[val_loader_1, val_loader_2],
        test_loader=test_loader_1,
        num_rounds=args.num_rounds,
        log_file=args.log_file,
        method=args.method,
        mu=args.mu
    )

if __name__ == "__main__":
    main()