import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import time
import os
import struct

# 表示設定
MAX_POINTS = 1000  # 表示するデータポイントの最大数（より滑らかな表示のため増加）

# シリアル通信の設定
SERIAL_PORT = 'COM6'  # Windowsの場合。必要に応じて変更してください
BAUD_RATE = 115200

# シリアルポートを開く
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

# プロット設定
WINDOW_SIZE = 3  # 表示する時間幅（秒）
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('IMUデータ')

# データバッファの初期化（dequeを使用）
timestamps = deque(maxlen=MAX_POINTS)
accel_data = [deque(maxlen=MAX_POINTS) for _ in range(3)]  # X, Y, Z
gyro_data = [deque(maxlen=MAX_POINTS) for _ in range(3)]   # X, Y, Z
temp_data = deque(maxlen=MAX_POINTS)

# グラフのタイトルとラベル
accel_titles = ['加速度 X', '加速度 Y', '加速度 Z']
gyro_titles = ['角速度 X', '角速度 Y', '角速度 Z']

# データ保存用ディレクトリの作成
if not os.path.exists('data'):
    os.makedirs('data')

print(f"シリアルポート {SERIAL_PORT} に接続しました")
start_time = time.time()
last_update_time = start_time

# グラフの初期化
lines = []

def init():
    for i in range(3):
        # 加速度グラフの設定
        axs[0, i].set_xlim(0, WINDOW_SIZE)
        axs[0, i].set_ylim(-20, 20)
        line, = axs[0, i].plot([], [], 'b-')
        lines.append(line)
        axs[0, i].set_title(accel_titles[i])
        axs[0, i].grid(True)
        if i == 0:
            axs[0, i].set_ylabel('加速度 (G)')

        # 角速度グラフの設定
        axs[1, i].set_xlim(0, WINDOW_SIZE)
        axs[1, i].set_ylim(-250/180*np.pi, 250/180*np.pi)  # 角速度範囲を拡大
        line, = axs[1, i].plot([], [], 'r-')
        lines.append(line)
        axs[1, i].set_title(gyro_titles[i])
        axs[1, i].set_xlabel('時間 (秒)')
        axs[1, i].grid(True)
        if i == 0:
            axs[1, i].set_ylabel('角速度 (dps)')

    plt.tight_layout()
    return lines

def update(frame):
    global last_update_time
    current_time = time.time()
    elapsed_time = current_time - start_time

    # シリアルデータの取得と処理
    try:
        available = ser.in_waiting
        if available > 1000:  # バッファが大きくなりすぎた場合
            ser.reset_input_buffer()  # 入力バッファをクリア
            print("バッファをクリアしました")
            return lines
            
        while ser.in_waiting >= 28:  # バッファ内のすべてのデータを処理
            data = ser.read(28)
            data_array = struct.unpack('7f', data)
            
            # データをdequeに追加（直近の時刻を使用）
            current_time = time.time()
            timestamps.append(current_time - start_time)
            for i in range(3):
                accel_data[i].append(data_array[i])
                gyro_data[i].append(data_array[i+3])
            temp_data.append(data_array[6])
            
    except Exception as e:
        print(f"エラー: {e}")

    # データの更新（十分なデータがある場合のみ更新）
    if len(timestamps) > 2:  # 最低2点以上のデータがある場合のみ更新
        # 一度だけnumpy配列に変換
        times = np.array(timestamps)
        x_min = max(0, times[-1] - WINDOW_SIZE)
        x_max = times[-1]
        
        # 表示範囲外の古いデータを削除
        cutoff_time = x_max - WINDOW_SIZE * 2  # 表示範囲の2倍の期間を保持
        while len(timestamps) > 0 and timestamps[0] < cutoff_time:
            timestamps.popleft()
            for i in range(3):
                accel_data[i].popleft()
                gyro_data[i].popleft()
            temp_data.popleft()
            
        # 表示範囲内のデータのみを使用
        mask = times >= x_min
        if np.any(mask):  # マスクされたデータが存在する場合のみ更新
            visible_times = times[mask]
            
            # X軸の範囲を更新
            for ax_row in axs:
                for ax in ax_row:
                    ax.set_xlim(x_min, x_max)
            
            # データ配列を一度だけ変換
            accel_arrays = [np.array(list(data))[mask] for data in accel_data]
            gyro_arrays = [np.array(list(data))[mask] for data in gyro_data]
            
            # データを一括更新
            for i in range(3):
                lines[i].set_data(visible_times, accel_arrays[i])
                lines[i+3].set_data(visible_times, gyro_arrays[i])

    return lines

ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=1, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    print("KeyboardInterrupt を検出")
finally:
    ser.close()
    print("スクリプトを終了しました。")
