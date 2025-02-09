import csv
import numpy as np
import torch
from model import IMUPredictor, IMULinearRegression

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

def main():
    model_name = "linear"
    # model_name = "transformer"
    if model_name == "linear":
        model = IMULinearRegression(num_classes=4, window_size=WINDOW_SIZE)
    else:
        model = IMUPredictor(num_classes=4)
    model.load(f"models/best_{model_name}_model.pth")
    model.eval()

    data_buffer = np.zeros((12, WINDOW_SIZE))
    labels_buffer = []
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
            labels_buffer.append(true_label)

            if idx >= WINDOW_SIZE - 1:
                features = np.array([preprocess_data(data_buffer[:, i]) for i in range(WINDOW_SIZE)])
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                with torch.no_grad():
                    output = model(features_tensor)
                predicted_idx = output.argmax().item()
                predicted_label = idx_to_label[predicted_idx]
                if predicted_label == labels_buffer[-1]:
                    correct += 1
                total += 1

            idx += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
