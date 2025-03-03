import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import cv2
import time
from collections import defaultdict

# UDPサーバーのIPアドレスとポート番号
UDP_IP = "0.0.0.0"
UDP_PORT = 8888

# ソケットの作成
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
labels = ["AccelNorm", "GyroNorm"]

# IDごとの色を定義
id_colors = {
    0: 'purple',
    1: 'blue',
    2: 'red',
    3: 'green',
}

# プロット設定
WINDOW_SIZE = 3  # 表示する時間幅（秒）
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax = axs[0]  # IMU data plot
ax_image = axs[1]  # Camera image plot

# データとラインを管理
id_data = np.full((4, 2, 0), np.nan, dtype=np.float32)  # (4 IDs, 2 values per ID, time points)
id_timestamps = np.array([])  # Single timestamp array for all IDs
id_lines = {}  # ID -> [accel_line, gyro_line]

ax_image_artist = None
print(f"Listening on {UDP_IP}:{UDP_PORT}")
video_frames = []
start_time = time.time()
last_update_time = start_time
current_time = start_time

# カメラの初期化
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: カメラがオープンできません")

def init():
    global ax_image_artist
    ax.set_xlim(0, WINDOW_SIZE)
    ax.set_ylim(0, 2)  # ノルムは正の値なので範囲を調整
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.grid(True)

    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image_artist = ax_image.imshow(np.zeros((480, 640, 3)))
    
    # 空のラインを返す（IDごとのラインは動的に追加される）
    return [ax_image_artist]

def update(frame):
    global id_data, id_timestamps, id_lines, video_frames, current_time, last_update_time
    current_time = time.time()
    elapsed_time = current_time - start_time
    # FPS計算（実際の更新間隔）
    update_interval = current_time - last_update_time
    last_update_time = current_time
    print(f"Update interval: {update_interval*1000:.1f}ms (FPS: {1/update_interval:.1f})")

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
    # 受信したデータを新しい列に設定
    for device_id, imu_data in latest_data.items():
        new_column[device_id, :, 0] = imu_data
        # IDに対応するラインがなければ作成
        if device_id not in id_lines:
            color = id_colors.get(device_id, f'C{device_id % 10}')
            accel_line, = ax.plot([], [], lw=2, label=f'ID{device_id}-Accel', color=color, linestyle='-')
            gyro_line, = ax.plot([], [], lw=2, label=f'ID{device_id}-Gyro', color=color, linestyle='--')
            id_lines[device_id] = [accel_line, gyro_line]
            ax.legend()

    # データを結合
    id_data = np.concatenate((id_data, new_column), axis=2)
    id_timestamps = np.append(id_timestamps, elapsed_time)

    # プロットの更新
    x_min = max(0, elapsed_time - WINDOW_SIZE)
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

    # カメラフレームの更新
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax_image_artist.set_data(frame_rgb)
        video_frames.append(frame)

    return all_lines + [ax_image_artist]

def save_data():
    global id_data, id_timestamps, video_frames, start_time
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # 全てNaN（無効な値）のIDを除外
    valid_ids = ~np.all(np.isnan(id_data), axis=(1,2))
    valid_id_indices = np.where(valid_ids)[0]
    filtered_id_data = id_data[valid_ids]
    print("valid IDs: ", valid_id_indices)

    # 各有効IDごとに有効な時間ポイントを特定
    valid_timepoints_per_id = []
    for id_idx in range(len(filtered_id_data)):
        # 現在のIDのデータについて、時間方向でNaNを含まないインデックスを取得
        id_valid_timepoints = ~np.any(np.isnan(filtered_id_data[id_idx]), axis=0)
        valid_timepoints_per_id.append(id_valid_timepoints)

    # 全ての有効IDで共通して有効な時間ポイントを取得
    if valid_timepoints_per_id:
        valid_timepoints = np.all(valid_timepoints_per_id, axis=0)
    else:
        valid_timepoints = np.zeros_like(id_timestamps, dtype=bool)

    final_data = filtered_id_data[:,:,valid_timepoints]
    final_timestamps = id_timestamps[valid_timepoints]
    print(f"valid timepoints: {final_timestamps.size}/{id_timestamps.size}")

    # データを保存
    data_dict = {
        "video_frames": np.array(video_frames) if video_frames else np.nan,
        "imu_data": final_data,
        "time_stamps": final_timestamps,
        "valid_ids": np.where(valid_ids)[0]  # 有効なIDのリスト
    }

    np.save(f"data/{timestamp}.npy", data_dict)
    print(f"Data saved to data/{timestamp}.npy")
    print(f"Total elapsed time: {time.time() - start_time:.2f} seconds")

ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=10, cache_frame_data=False) # interval=0にすると30FPSくらいになる

try:
    plt.show()
except KeyboardInterrupt:
    print("KeyboardInterrupt detected")
finally:
    save_data()
    cap.release()
    sock.close()
    print("Script finished.")
