import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from scripts.dataset import KeypointsSequenceDataset
from scripts.model import GAT_LSTM_Attn_Model
from scripts.util import save_model, save_confusion_matrix

def get_label_map(mode):
    if mode == "bin":
        return {
            'Falling forward using hands': 0,
            'Falling forward using knees': 0,
            'Falling backward': 0,
            'Falling sideways': 0,
            'Falling sitting in empty chair': 0,
            'Walking': 1,
            'Standing': 1,
            'Sitting': 1,
            'Picking up an object': 1,
            'Jumping': 1,
            'Laying': 1
        }
    elif mode == "mul":
        return {
            'Falling forward using hands': 0,
            'Falling forward using knees': 1,
            'Falling backward': 2,
            'Falling sideways': 3,
            'Falling sitting in empty chair': 4,
            'Walking': 5,
            'Standing': 6,
            'Sitting': 7,
            'Picking up an object': 8,
            'Jumping': 9,
            'Laying': 10
        }
    else:
        raise ValueError("Invalid mode. Choose 'bin' or 'mul'.")

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        batch_size, seq_len, num_nodes, in_channels = sequences.size()

        optimizer.zero_grad()
        edge_index = torch.tensor([
            [0, 1], [0, 2], [1, 3], [2, 4],
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            [5, 11], [6, 12], [11, 12],
            [11, 13], [12, 14], [13, 15], [14, 16]
        ], dtype=torch.long).t().contiguous().to(device)

        outputs = model(sequences, edge_index)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            batch_size, seq_len, num_nodes, in_channels = sequences.size()

            edge_index = torch.tensor([
                [0, 1], [0, 2], [1, 3], [2, 4],
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                [5, 11], [6, 12], [11, 12],
                [11, 13], [12, 14], [13, 15], [14, 16]
            ], dtype=torch.long).t().contiguous().to(device)

            outputs = model(sequences, edge_index)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    return epoch_loss, epoch_acc, precision, recall, f1, all_labels, all_predictions

def main():
    parser = argparse.ArgumentParser(description="Train GAT-LSTM model")
    parser.add_argument("--mode", type=str, choices=["bin", "mul"], default="mul", help="Classification mode: 'bin' or 'mul'")
    parser.add_argument("--data_paths", nargs="+", default=["./UP-FALL-output-processed2/Camera1", "./UP-FALL-output-processed2/Camera2"], help="Paths to data folders")
    parser.add_argument("--sequence_length", type=int, default=40, help="Sequence length for the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    label_map = get_label_map(args.mode)

    dataset = KeypointsSequenceDataset(args.data_paths, args.sequence_length, label_map)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.mode == 'bin':
        model = GAT_LSTM_Attn_Model(sequence_length=args.sequence_length, in_channels=3, hidden_channels=64, out_channels=128, num_classes=2).to(device)
    else:
        model = GAT_LSTM_Attn_Model(sequence_length=args.sequence_length, in_channels=3, hidden_channels=64, out_channels=128, num_classes=len(label_map)).to(device)
        
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(args.num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}')

        scheduler.step(val_loss)

    test_loss, test_acc, test_precision, test_recall, test_f1, test_labels, test_predictions = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    print(f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}')

    conf_matrix = confusion_matrix(test_labels, test_predictions)
    save_confusion_matrix(conf_matrix, list(label_map.keys()), os.path.join(args.output_dir, "confusion_matrix.png"))
    save_model(model, os.path.join(args.output_dir, "model.pth"))

if __name__ == "__main__":
    main()