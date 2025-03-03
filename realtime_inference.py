import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import cv2
import time
import torch
from model import IMUPredictor
from collections import defaultdict

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

# IDごとの色を定義
id_colors = {
    1: 'blue',
    2: 'red',
    3: 'green',
    4: 'purple',
    5: 'orange',
    # 必要に応じて追加
}

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Initialize plot
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax = axs[0]  # IMU data plot
ax_image = axs[1]  # Camera image plot
ax_image_artist = None

# IDごとのデータとラインを管理
id_data_buffer = defaultdict(lambda: np.zeros((2, WINDOW_SIZE)))  # ID -> (2 features x window size)
id_data_list = defaultdict(lambda: np.zeros((2, 0)))  # ID -> (2 features x time points)
id_current_idx = defaultdict(int)
id_timestamps = defaultdict(lambda: np.array([]))
id_lines = {}  # ID -> [accel_line, gyro_line]
id_predictions = {}  # ID -> predicted label

start_time = time.time()
last_update_time = start_time

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: カメラがオープンできません")

# Initialize model
model = IMUPredictor(num_classes=4)
model.load("models/best_model.pth")  # IMUPredictorのloadメソッドを使用
model.eval()

def preprocess_data(data):
    # データはすでにノルム（大きさ）になっているので、スケーリングのみ行う
    accel_norm = data[0] / ACCEL_SCALE
    gyro_norm = data[1] / GYRO_SCALE
    
    return np.array([accel_norm, gyro_norm])

def init():
    global ax_image_artist
    ax.set_xlim(0, 3)  # 3秒間のデータを表示
    ax.set_ylim(0, 50000)  # ノルムは正の値なので範囲を調整
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.grid(True)

    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image_artist = ax_image.imshow(np.zeros((480, 640, 3)))
    
    # 空のラインを返す（IDごとのラインは動的に追加される）
    return [ax_image_artist]

def update(frame):
    global id_data_buffer, id_data_list, id_current_idx, id_timestamps, id_lines, id_predictions, last_update_time
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
        data_array = np.frombuffer(data, dtype=np.float32)
        
        # 最初の要素がID、残りがIMUデータ
        device_id = int(data_array[0])
        imu_data = data_array[1:3]  # AccelNorm, GyroNorm
        
        # IDごとにデータバッファを更新
        id_data_buffer[device_id] = np.roll(id_data_buffer[device_id], -1, axis=1)
        id_data_buffer[device_id][:, -1] = imu_data
        
        # IDごとにデータリストを更新
        id_data_list[device_id] = np.hstack((id_data_list[device_id], imu_data.reshape(-1, 1)))
        id_timestamps[device_id] = np.append(id_timestamps[device_id], elapsed_time)
        
        # IDに対応するラインがなければ作成
        if device_id not in id_lines:
            color = id_colors.get(device_id, f'C{device_id % 10}')  # 定義されていないIDには自動的に色を割り当て
            accel_line, = ax.plot([], [], lw=2, label=f'ID{device_id}-Accel', color=color, linestyle='-')
            gyro_line, = ax.plot([], [], lw=2, label=f'ID{device_id}-Gyro', color=color, linestyle='--')
            id_lines[device_id] = [accel_line, gyro_line]
            ax.legend()
        
        # IDごとにカウンタを更新
        id_current_idx[device_id] += 1
        
        # データの前処理と推論
        if id_current_idx[device_id] >= WINDOW_SIZE:
            features = np.array([preprocess_data(id_data_buffer[device_id][:, i]) for i in range(WINDOW_SIZE)])
            # (batch_size, seq_len, feature_dim)の形式に変換
            features_tensor = torch.FloatTensor(features).unsqueeze(0).contiguous()  # Add batch dimension
            
            with torch.no_grad():
                output = model(features_tensor)
                predicted_idx = output.argmax().item()
                predicted_label = idx_to_label[predicted_idx]
                id_predictions[device_id] = predicted_label

    # プロットの更新
    x_min = max(0, elapsed_time - 3)
    x_max = elapsed_time
    ax.set_xlim(x_min, x_max)
    
    all_lines = []
    for device_id, lines in id_lines.items():
        if len(id_timestamps[device_id]) > 0:
            # 表示範囲内のデータのみを使用
            mask = (id_timestamps[device_id] >= x_min) & (id_timestamps[device_id] <= x_max)
            visible_times = id_timestamps[device_id][mask]
            visible_data = id_data_list[device_id][:, mask]
            
            # データの更新
            lines[0].set_data(visible_times, visible_data[0])  # AccelNorm
            lines[1].set_data(visible_times, visible_data[1])  # GyroNorm
            
            all_lines.extend(lines)

    # カメラフレームの更新と推論結果の表示
    ret, frame = cap.read()
    if ret:
        # 各IDの推論結果をフレームに描画
        y_pos = 30
        for device_id, prediction in id_predictions.items():
            color = id_colors.get(device_id, (0, 0, 255))  # デフォルトは赤
            if isinstance(color, str):
                # 文字列の色名をRGB値に変換
                if color == 'blue': bgr_color = (255, 0, 0)
                elif color == 'red': bgr_color = (0, 0, 255)
                elif color == 'green': bgr_color = (0, 255, 0)
                elif color == 'purple': bgr_color = (255, 0, 255)
                elif color == 'orange': bgr_color = (0, 165, 255)
                else: bgr_color = (0, 0, 255)  # デフォルトは赤
            else:
                bgr_color = color
                
            cv2.putText(
                frame,
                f"ID{device_id}: {prediction}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                bgr_color,
                2,
                cv2.LINE_AA
            )
            y_pos += 30
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax_image_artist.set_data(frame_rgb)

    return all_lines + [ax_image_artist]

ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=10, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    print("KeyboardInterrupt detected")
finally:
    cap.release()
    sock.close()
    print("Script finished.")
