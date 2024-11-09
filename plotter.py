import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# UDPサーバーのIPアドレスとポート番号
UDP_IP = "0.0.0.0"
UDP_PORT = 8888

# ソケットの作成
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
lavels = ["Acc1X", "Acc1Y", "Acc1Z", "Gyro1X", "Gyro1Y", "Gyro1Z", "Acc2X", "Acc2Y", "Acc2Z", "Gyro2X", "Gyro2Y", "Gyro2Z"]

fig, ax = plt.subplots()
lines = [ax.plot([], [], lw=2, label=label)[0] for label in lavels]

print(f"Listening on {UDP_IP}:{UDP_PORT}")

data_list = np.zeros((12, 0))

def init():
    ax.set_xlim(0, 100)
    ax.set_ylim(-30000, 30000)
    # ラベルの設定
    ax.set_xlabel("Frame")
    ax.set_ylabel("Value")
    ax.legend()
    return lines

def update(frame):
    global data_list
    data = None
    while True: # 最新のデータを取得
        try:
            sock.settimeout(0.01)
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            break
    if data is not None:
        data = np.frombuffer(data, dtype=np.float32)
        print(data)
        data_list = np.hstack((data_list, data.reshape(-1, 1)))
        if data_list.shape[1] > 100:
            data_list = data_list[:, 1:]
        for i, line in enumerate(lines):
            line.set_data(np.arange(data_list.shape[1]), data_list[i])
    return lines

ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=10) # 10msごとに描画

plt.show()