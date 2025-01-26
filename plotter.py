import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import cv2
import time

# UDPサーバーのIPアドレスとポート番号
UDP_IP = "0.0.0.0"
UDP_PORT = 8888

# ソケットの作成
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
lavels = ["Acc1X", "Acc1Y", "Acc1Z", "Gyro1X", "Gyro1Y", "Gyro1Z", "Acc2X", "Acc2Y", "Acc2Z", "Gyro2X", "Gyro2Y", "Gyro2Z"]

# プロット設定
WINDOW_SIZE = 10  # 表示する時間幅（秒）
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax = axs[0]  # IMU data plot
ax_image = axs[1]  # Camera image plot
lines = [ax.plot([], [], lw=2, label=label)[0] for label in lavels]
ax_image_artist = None
print(f"Listening on {UDP_IP}:{UDP_PORT}")

# データ保存用の配列（特徴量×時間の形式）
data_list = np.zeros((12, 0))
time_stamps = np.array([])
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
    ax.set_ylim(-30000, 30000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image_artist = ax_image.imshow(np.zeros((480, 640, 3)))
    return lines + [ax_image_artist]

def update(frame):
    global data_list, time_stamps, video_frames, current_time, last_update_time
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    # FPS計算（実際の更新間隔）
    update_interval = current_time - last_update_time
    last_update_time = current_time
    print(f"Update interval: {update_interval*1000:.1f}ms (FPS: {1/update_interval:.1f})")

    # UDPデータの取得
    data = None
    while True:
        try:
            sock.settimeout(0.001)  # より短いタイムアウトで最新データを取得
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            break

    if data is not None:
        # 新しいデータの追加
        data = np.frombuffer(data, dtype=np.float32)
        data_list = np.hstack((data_list, data.reshape(-1, 1)))
        time_stamps = np.append(time_stamps, elapsed_time)

    # プロットの更新
    if len(time_stamps) > 0:
        # 表示範囲の設定
        x_min = max(0, elapsed_time - WINDOW_SIZE)
        x_max = elapsed_time
        ax.set_xlim(x_min, x_max)

        # 表示範囲内のデータのみを使用
        mask = (time_stamps >= x_min) & (time_stamps <= x_max)
        visible_times = time_stamps[mask]
        visible_data = data_list[:, mask]

        # データの更新
        for i, line in enumerate(lines):
            line.set_data(visible_times, visible_data[i])

    # カメラフレームの更新
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax_image_artist.set_data(frame_rgb)
        video_frames.append(frame)

    return lines + [ax_image_artist]

def save_data():
    global data_list, video_frames, start_time, time_stamps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    data_dict = {
        "imu_data": data_list,
        "time_stamps": time_stamps,
        "video_frames": np.array(video_frames) if video_frames else None
    }
    np.save(f"data/{timestamp}.npy", data_dict)
    print(f"Data saved to data/{timestamp}.npy")
    print(f"Total elapsed time: {time.time() - start_time:.2f} seconds")
    print(f"Total data points: {len(time_stamps)}")

# アニメーションの設定（33ms ≈ 30fps）
ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=33)

try:
    plt.show()
except KeyboardInterrupt:
    print("KeyboardInterrupt detected, saving data...")
    save_data()
finally:
    cap.release()
    sock.close()
    print("Script finished.")
