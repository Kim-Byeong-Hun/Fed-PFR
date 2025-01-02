import torch
import matplotlib.pyplot as plt
import seaborn as sns

def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_confusion_matrix(cm, labels, path):
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(path)
    plt.close()