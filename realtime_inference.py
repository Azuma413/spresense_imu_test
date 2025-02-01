import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import cv2
import time
import torch
from model import IMUPredictor

# Constants
UDP_IP = "0.0.0.0"
UDP_PORT = 8888
WINDOW_SIZE = 60  # モデルの入力ウィンドウサイズ
ACCEL_SCALE = 2000
GYRO_SCALE = 300000

# Label mapping
idx_to_label = {
    0: 'run',
    1: 'walk',
    2: 'shake',
    3: 'something'
}

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Initialize plot
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax = axs[0]  # IMU data plot
ax_image = axs[1]  # Camera image plot
labels = ["Acc1X", "Acc1Y", "Acc1Z", "Gyro1X", "Gyro1Y", "Gyro1Z", 
          "Acc2X", "Acc2Y", "Acc2Z", "Gyro2X", "Gyro2Y", "Gyro2Z"]
lines = [ax.plot([], [], lw=2, label=label)[0] for label in labels]
ax_image_artist = None

# Initialize data storage
data_buffer = np.zeros((12, WINDOW_SIZE))  # 12 features x window size for model
data_list = np.zeros((12, 0))  # For plotting all historical data
current_idx = 0
time_stamps = np.array([])
start_time = time.time()
last_update_time = start_time

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: カメラがオープンできません")

# Initialize model
model = IMUPredictor(num_classes=4)
model.load("best_model.pth")  # IMUPredictorのloadメソッドを使用
model.eval()

def preprocess_data(data):
    # Calculate norms for both IMUs
    accel1_norm = np.sqrt(data[0]**2 + data[1]**2 + data[2]**2)
    accel2_norm = np.sqrt(data[6]**2 + data[7]**2 + data[8]**2)
    gyro1_norm = np.sqrt(data[3]**2 + data[4]**2 + data[5]**2)
    gyro2_norm = np.sqrt(data[9]**2 + data[10]**2 + data[11]**2)
    
    # Average norms and scale
    accel_norm = (accel1_norm + accel2_norm) / (2 * ACCEL_SCALE)
    gyro_norm = (gyro1_norm + gyro2_norm) / (2 * GYRO_SCALE)
    
    return np.array([accel_norm, gyro_norm])

def init():
    global ax_image_artist
    ax.set_xlim(0, 3)  # 3秒間のデータを表示
    ax.set_ylim(-200000, 200000)  # より広いY軸範囲を設定
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image_artist = ax_image.imshow(np.zeros((480, 640, 3)))
    return lines + [ax_image_artist]

def update(frame):
    global data_buffer, data_list, current_idx, time_stamps, last_update_time
    current_time = time.time()
    elapsed_time = current_time - start_time
    update_interval = current_time - last_update_time
    last_update_time = current_time

    # UDPデータの取得
    data = None
    while True:
        try:
            sock.settimeout(0.001)
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            break

    if data is not None:
        # データの更新
        raw_data = np.frombuffer(data, dtype=np.float32)
        # Update data_buffer for model inference
        data_buffer = np.roll(data_buffer, -1, axis=1)
        data_buffer[:, -1] = raw_data
        # Update data_list for plotting
        data_list = np.hstack((data_list, raw_data.reshape(-1, 1)))
        time_stamps = np.append(time_stamps, elapsed_time)

        # プロットの更新
        x_min = max(0, elapsed_time - 3)
        x_max = elapsed_time
        ax.set_xlim(x_min, x_max)
        mask = (time_stamps >= x_min) & (time_stamps <= x_max)
        visible_times = time_stamps[mask]
        visible_data = data_list[:, mask]

        for i, line in enumerate(lines):
            line.set_data(visible_times, visible_data[i])

        # データの前処理と推論
        if current_idx >= WINDOW_SIZE - 1:
            features = np.array([preprocess_data(data_buffer[:, i]) for i in range(WINDOW_SIZE)])
            # (batch_size, seq_len, feature_dim)の形式に変換
            features_tensor = torch.FloatTensor(features).unsqueeze(0).contiguous()  # Add batch dimension
            
            with torch.no_grad():
                output = model(features_tensor)
                predicted_idx = output.argmax().item()
                predicted_label = idx_to_label[predicted_idx]

    # カメラフレームの更新と推論結果の表示
    ret, frame = cap.read()
    if ret:
        # 推論結果をフレームに描画
        if data is not None and current_idx >= WINDOW_SIZE - 1:
            cv2.putText(
                frame,
                f"Prediction: {predicted_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax_image_artist.set_data(frame_rgb)

    current_idx += 1
    return lines + [ax_image_artist]

ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=10, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    print("KeyboardInterrupt detected")
finally:
    cap.release()
    sock.close()
    print("Script finished.")
