import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import cv2
import time
import torch
from model import IMUPredictor, IMUConvNet, IMULinearRegression
from collections import defaultdict

# Constants
UDP_IP = "0.0.0.0"
UDP_PORT = 8888
WINDOW_SIZE = 30  # eval.pyと同じウィンドウサイズ
FEATURE_DIM = 2*4  # eval.pyと同じ特徴量次元
NUM_CLASSES = 3

# Label mapping (eval.pyと同じラベル)
idx_to_label = {
    0: 'run',
    1: 'walk',
    2: 'something'
}

# IDごとの色を定義
id_colors = {
    0: 'purple',
    1: 'blue',
    2: 'red',
    3: 'green',
}

# Global flag for legend creation
legend_created = False

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Initialize plot
# Create figure with proper spacing
fig = plt.figure(figsize=(12, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Create grid layout
gs = plt.GridSpec(2, 2)

# Create subplots with adjusted positions
ax = fig.add_subplot(gs[0, 0])  # IMU data plot
ax_image = fig.add_subplot(gs[0, 1])  # Camera image plot
ax_calories = fig.add_subplot(gs[1, :])  # Calories plot spanning both columns

ax.set_title('IMU Data')
ax_image.set_title('Camera Feed')
ax_calories.set_title('Calories Burned (per 10 seconds)')
ax_image_artist = None

# Calorie calculation constants
METS = {'run': 6.0, 'walk': 3.0, 'something': 1.5}
BODY_WEIGHT = 60  # Default weight in kg
CALORIE_WINDOW = 60  # 10分 (60 steps of 10-second intervals)
accumulated_calories = 0  # 10秒間の累積カロリー
calorie_line = None
calorie_times = np.arange(CALORIE_WINDOW)
calorie_data = np.zeros(CALORIE_WINDOW)

# plotter.pyと同様のデータ構造
id_data = np.full((4, 2, 0), np.nan, dtype=np.float32)  # (4 IDs, 2 values per ID, time points)
id_timestamps = np.array([])  # Single timestamp array for all IDs
id_lines = {}  # ID -> [accel_line, gyro_line]
current_prediction = None  # 現在の予測結果（全IMUで共通）
id_data_buffer = defaultdict(lambda: np.full((2, WINDOW_SIZE), np.nan))  # Window buffer for inference

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize timers
start_time = time.time()
last_calorie_time = time.time()
last_update_time = start_time

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: カメラがオープンできません")

# Initialize model (eval.pyと同様のモデル初期化)
model_name = "conv"  # conv, linear, transformer から選択
if model_name == "transformer":
    model = IMUPredictor(num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM).to(DEVICE)
elif model_name == "conv":
    model = IMUConvNet(num_classes=NUM_CLASSES, window_size=WINDOW_SIZE, feature_dim=FEATURE_DIM).to(DEVICE)
elif model_name == "linear":
    model = IMULinearRegression(num_classes=NUM_CLASSES, window_size=WINDOW_SIZE).to(DEVICE)
else:
    raise ValueError(f"Invalid model name: {model_name}")

model.load(f"models/best_{model_name}_model.pth")
model.eval()

def preprocess_data(all_data_buffer):
    """
    all_data_buffer: shape (4, 2, WINDOW_SIZE) - (num_imus, features_per_imu, window_size)
    returns: shape (WINDOW_SIZE, FEATURE_DIM) where FEATURE_DIM = 8 (2 features × 4 IMUs)
    """
    features = []
    for i in range(WINDOW_SIZE):
        # Combine features from all IMUs for this time step
        time_step_features = []
        for imu_id in range(4):  # 4つのIMU
            if imu_id in all_data_buffer and not np.any(np.isnan(all_data_buffer[imu_id][:, i])):
                # Use the features only if they are not NaN
                accel_norm = all_data_buffer[imu_id][0, i]
                gyro_norm = all_data_buffer[imu_id][1, i]
                time_step_features.extend([accel_norm, gyro_norm])
            else:
                # If IMU data is missing or contains NaN, pad with zeros
                time_step_features.extend([0, 0])
        features.append(time_step_features)
    return np.array(features)  # shape: (WINDOW_SIZE, 8)

def init():
    global ax_image_artist, calorie_line, legend_created
    ax.set_xlim(0, 3)  # 3秒間のデータを表示
    ax.set_ylim(0, 2)  # ノルムは正の値なので範囲を調整
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.grid(True)

    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image_artist = ax_image.imshow(np.zeros((480, 640, 3)))
    
    # Initialize calories plot
    ax_calories.set_xlim(0, CALORIE_WINDOW)
    ax_calories.set_ylim(0, 5)  # Initial range, will be adjusted dynamically
    ax_calories.set_xlabel("Time (steps of 10 seconds)")
    ax_calories.set_ylabel("Calories")
    ax_calories.grid(True)
    
    # Create line with label only for the first time
    if not legend_created:
        calorie_line, = ax_calories.plot([], [], 'r-', label='Calories/10s', linewidth=2)
        ax_calories.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        legend_created = True
    else:
        calorie_line, = ax_calories.plot([], [], 'r-', label='_nolegend_', linewidth=2)
    
    # Adjust layout to prevent label overlap
    plt.tight_layout()
    return [ax_image_artist, calorie_line]

def update(frame):
    global id_data, id_timestamps, id_lines, current_prediction, id_data_buffer, \
           calorie_data, calorie_times, last_calorie_time, accumulated_calories, \
           last_update_time

    current_time = time.time()
    elapsed_time = current_time - start_time
    update_interval = current_time - last_update_time
    last_update_time = current_time

    # UDPデータの取得とデータの更新
    latest_data = {}  # ID毎の最新データを保持する辞書
    while True:
        try:
            sock.settimeout(0.001)  # より短いタイムアウトで最新データを取得
            data, addr = sock.recvfrom(1024)
            # 3つのfloat値として受信
            data_array = np.frombuffer(data, dtype=np.float32)
            if len(data_array) >= 3:
                # floatとして送信されたIDをintに変換
                device_id = int(round(data_array[0]))  # round to handle potential float imprecision
                if 0 <= device_id <= 3:  # ID 0-3のデータのみを処理
                    imu_data = data_array[1:3]  # AccelNorm, GyroNorm
                    latest_data[device_id] = imu_data  # 最新のデータを保存
        except socket.timeout:
            break

    new_column = np.full((4, 2, 1), np.nan, dtype=np.float32)
    has_new_data = False
    # 受信したデータを新しい列に設定
    for device_id, imu_data in latest_data.items():
        new_column[device_id, :, 0] = imu_data
        has_new_data = True
        
        # IDごとにデータバッファを更新（推論用）
        id_data_buffer[device_id] = np.roll(id_data_buffer[device_id], -1, axis=1)
        id_data_buffer[device_id][:, -1] = imu_data

    # 新しいデータがある場合のみ処理を続行
    if has_new_data:
        # 各IDのラインを作成/更新
        for device_id in latest_data.keys():
            if device_id not in id_lines:
                color = id_colors.get(device_id, f'C{device_id % 10}')
                accel_line, = ax.plot([], [], lw=2, label=f'ID{device_id}-Accel', color=color, linestyle='-')
                gyro_line, = ax.plot([], [], lw=2, label=f'ID{device_id}-Gyro', color=color, linestyle='--')
                id_lines[device_id] = [accel_line, gyro_line]
                ax.legend()

        # すべてのIMU (0-3) からデータを受信したか確認
        all_imus_present = all(i in latest_data for i in range(4))
        
        # バッファに十分なデータが蓄積されているか確認
        has_enough_data = all(np.any(~np.isnan(id_data_buffer[i])) for i in range(4))
        
        # すべてのIMUからデータを受信し、かつバッファに十分なデータがある場合のみ推論を実行
        if all_imus_present and has_enough_data:
            # 全IMUのデータを組み合わせて特徴量を作成
            features = preprocess_data(id_data_buffer)
            # (batch_size, seq_len, feature_dim)の形式に変換
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(features_tensor)
                predicted_idx = output.argmax().item()
                if predicted_idx in idx_to_label:
                    current_prediction = idx_to_label[predicted_idx]

    # データを結合
    id_data = np.concatenate((id_data, new_column), axis=2)
    id_timestamps = np.append(id_timestamps, elapsed_time)

    # プロットの更新
    x_min = max(0, elapsed_time - 3)
    x_max = elapsed_time
    ax.set_xlim(x_min, x_max)

    all_lines = []
    # 表示範囲内のデータのマスク
    mask = (id_timestamps >= x_min) & (id_timestamps <= x_max)
    visible_times = id_timestamps[mask]
    visible_data = id_data[:, :, mask]

    for device_id, lines in id_lines.items():
        # 有効なデータが存在する場合のみプロット
        if ~np.all(np.isnan(visible_data[device_id])):
            device_data = visible_data[device_id]
            # NaNを含まない時間のインデックスを取得
            valid_mask = ~np.any(np.isnan(device_data), axis=0)
            if np.any(valid_mask):
                lines[0].set_data(visible_times[valid_mask], device_data[0, valid_mask])  # AccelNorm
                lines[1].set_data(visible_times[valid_mask], device_data[1, valid_mask])  # GyroNorm
                all_lines.extend(lines)

    # カメラフレームの更新と推論結果の表示
    ret, frame = cap.read()
    if ret:
        # 推論結果の描画（全IMUで共通）
        if current_prediction:
            cv2.putText(
                frame,
                f"Prediction: {current_prediction}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),  # 緑色で表示
                2,
                cv2.LINE_AA
            )
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax_image_artist.set_data(frame_rgb)

    if current_prediction in METS:
        # cal = METs * time(s) * weight(kg) * 1.05 / 3600
        interval_calories = METS[current_prediction] * update_interval * BODY_WEIGHT * 1.05 / 3600
        accumulated_calories += interval_calories

    if current_time - last_calorie_time >= 10:  # 10秒ごとに更新
        # 実際の経過時間を記録
        elapsed_time_in_minutes = (current_time - start_time) / 60  # 分単位
        # Update calorie data with rolling window
        calorie_data = np.roll(calorie_data, -1)
        calorie_data[-1] = accumulated_calories  # 10秒間の消費カロリー
        # 時間軸も実際の経過時間で更新
        calorie_times = np.roll(calorie_times, -1)
        calorie_times[-1] = elapsed_time_in_minutes
        # グラフの表示範囲を調整（直近のデータを表示）
        if elapsed_time_in_minutes > CALORIE_WINDOW/6:  # 10分以上経過したら表示範囲をスクロール
            ax_calories.set_xlim(elapsed_time_in_minutes - CALORIE_WINDOW/6, elapsed_time_in_minutes)
        accumulated_calories = 0
        last_calorie_time = current_time

    # Update calorie line data
    calorie_line.set_data(calorie_times, calorie_data)
    
    # Dynamically adjust y-axis based on actual calorie values
    if len(calorie_data[calorie_data > 0]) > 0:
        max_calories = np.max(calorie_data[calorie_data > 0])
        current_ymax = ax_calories.get_ylim()[1]
        if max_calories > current_ymax * 0.8:
            ax_calories.set_ylim(0, max_calories * 1.2)
        elif max_calories < current_ymax * 0.3:
            ax_calories.set_ylim(0, max_calories * 1.5)
    
    # Adjust x-axis ticks for better readability
    ax_calories.xaxis.set_major_locator(plt.MaxNLocator(10))
    return all_lines + [ax_image_artist, calorie_line]

ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=10, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    print("KeyboardInterrupt detected")
finally:
    cap.release()
    sock.close()
    print("Script finished.")
