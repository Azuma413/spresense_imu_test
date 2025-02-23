import csv
import numpy as np
import torch
from model import IMUPredictor, IMULinearRegression, IMUConvNet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

WINDOW_SIZE = 60
ACCEL_SCALE = 2000
GYRO_SCALE = 300000

idx_to_label = {
    0: 'run',
    1: 'walk',
    2: 'shake',
    3: 'something'
}

def preprocess_data(data):
    accel1_norm = np.sqrt(data[0]**2 + data[1]**2 + data[2]**2)
    accel2_norm = np.sqrt(data[6]**2 + data[7]**2 + data[8]**2)
    gyro1_norm = np.sqrt(data[3]**2 + data[4]**2 + data[5]**2)
    gyro2_norm = np.sqrt(data[9]**2 + data[10]**2 + data[11]**2)
    accel_norm = (accel1_norm + accel2_norm) / (2 * ACCEL_SCALE)
    gyro_norm = (gyro1_norm + gyro2_norm) / (2 * GYRO_SCALE)
    return np.array([accel_norm, gyro_norm])

def main(model_name):
    if model_name == "linear":
        model = IMULinearRegression(num_classes=4, window_size=WINDOW_SIZE)
    elif model_name == "transformer":
        model = IMUPredictor(num_classes=4)
    elif model_name == "conv":
        model = IMUConvNet(num_classes=4, window_size=WINDOW_SIZE)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    model.load(f"models/best_{model_name}_model.pth")
    # パラメータ数を表示
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.eval()

    data_buffer = np.zeros((12, WINDOW_SIZE))
    labels_buffer = []
    predicted_labels = []
    correct = 0
    total = 0
    idx = 0

    with open('data/eval_labels.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # First 12 columns: sensor data; last column: label
            # 1行目はヘッダーなのでスキップ
            if idx == 0:
                idx += 1
                continue
            raw_data = np.array(list(map(float, row[:12])), dtype=np.float32)
            true_label = row[12].strip()

            data_buffer = np.roll(data_buffer, -1, axis=1)
            data_buffer[:, -1] = raw_data

            if idx >= WINDOW_SIZE - 1:
                features = np.array([preprocess_data(data_buffer[:, i]) for i in range(WINDOW_SIZE)])
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                with torch.no_grad():
                    output = model(features_tensor)
                predicted_idx = output.argmax().item()
                predicted_label = idx_to_label[predicted_idx]
                predicted_labels.append(predicted_label)
                labels_buffer.append(true_label)
                if predicted_label == true_label:
                    correct += 1
                total += 1

            idx += 1

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
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()
    print(f"Confusion matrix has been saved as 'confusion_matrix_{model_name}.png'")

if __name__ == '__main__':
    model_name = "linear"
    # model_name = "transformer"
    # model_name = "conv"
    main(model_name)
