import numpy as np
import torch
from model import IMUPredictor, IMULinearRegression, IMUConvNet, IMUDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

WINDOW_SIZE = 30
FEATURE_DIM = 2*4
NUM_CLASSES = 3

def main(model_name):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize model
    if model_name == "transformer":
        model = IMUPredictor(num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM).to(DEVICE)
    elif model_name == "conv":
        model = IMUConvNet(num_classes=NUM_CLASSES, window_size=WINDOW_SIZE, feature_dim=FEATURE_DIM).to(DEVICE)
    elif model_name == "linear":
        model = IMULinearRegression(num_classes=NUM_CLASSES, window_size=WINDOW_SIZE, num_features=FEATURE_DIM).to(DEVICE)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model.load(f"models/best_{model_name}_model.pth")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.eval()

    # Initialize dataset
    dataset = IMUDataset(['data/eval_labels.csv'], window_size=WINDOW_SIZE)
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}

    correct = 0
    total = 0
    labels_buffer = []
    predicted_labels = []

    # Evaluation loop
    for i in range(len(dataset)):
        features, true_label = dataset[i]
        features = features.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to correct device

        with torch.no_grad():
            output = model(features)
            predicted_idx = output.argmax().item()
            true_label_str = idx_to_label[true_label.item()]
            predicted_label_str = idx_to_label[predicted_idx]
            labels_buffer.append(true_label_str)
            predicted_labels.append(predicted_label_str)
            if predicted_idx == true_label.item():
                correct += 1
            total += 1

    # Calculate and print accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f}")
    # Calculate confusion matrix
    labels = list(idx_to_label.values())
    cm = confusion_matrix(labels_buffer, predicted_labels, labels=labels)
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name} model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'images/confusion_matrix_{model_name}.png')
    plt.close()
    print(f"Confusion matrix has been saved as 'confusion_matrix_{model_name}.png'")

if __name__ == '__main__':
    # model_name = "linear"
    model_name = "transformer"
    # model_name = "conv"
    main(model_name)
