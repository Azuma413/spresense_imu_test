import serial
import struct
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # バックエンドを明示的に設定
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class PosturePlotter:
    def __init__(self):
        # シリアルポートの設定
        self.ser = serial.Serial('COM6', 115200)  # Windowsの場合。必要に応じてポートを変更してください
        print(f"シリアルポート COM6 に接続しました")
        
        # 3Dプロットの初期設定
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_zlim([-1.5, 1.5])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.grid(True)
        
        # 矢印の初期化
        self.arrows = []
        self.arrow_length = 1.0
        
        # 基準座標系の単位ベクトル
        self.base_vectors = np.eye(3)
        
        # 初期状態の矢印を描画
        self.draw_arrows(np.eye(3))
        
    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """オイラー角から回転行列を計算"""
        # ロール(X軸周り)の回転行列
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # ピッチ(Y軸周り)の回転行列
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # ヨー(Z軸周り)の回転行列
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # 全体の回転行列 (R = Rz * Ry * Rx)
        return Rz @ Ry @ Rx

    def draw_arrows(self, R):
        """回転行列に基づいて矢印を描画"""
        # 以前の矢印を削除
        for arrow in self.arrows:
            arrow.remove()
        self.arrows.clear()
        
        # 3つの軸の矢印を描画
        colors = ['r', 'g', 'b']  # x軸:赤, y軸:緑, z軸:青
        for i in range(3):
            rotated_vec = R @ self.base_vectors[i] * self.arrow_length
            arrow = self.ax.quiver(0, 0, 0,
                                 rotated_vec[0], rotated_vec[1], rotated_vec[2],
                                 color=colors[i], alpha=0.8, linewidth=2)
            self.arrows.append(arrow)
        
        # プロットの更新
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_plot(self, frame):
        """プロットの更新関数"""
        try:
            # バッファサイズのチェック
            available = self.ser.in_waiting
            if available > 1000:  # バッファが大きくなりすぎた場合
                self.ser.reset_input_buffer()  # 入力バッファをクリア
                print("バッファをクリアしました")
                return
            
            # パケットの読み取りと解析
            if self.ser.in_waiting >= 15:  # ヘッダー(1) + 長さ(1) + データ(12) + チェックサム(1) + フッター(1)
                # ヘッダーの確認
                header = self.ser.read(1)[0]
                if header != 0xAA:
                    print("無効なヘッダー")
                    return

                # データ長の読み取り
                length = self.ser.read(1)[0]
                if length != 12:  # float型3個分のバイト数
                    print(f"無効なデータ長: {length}")
                    return

                # データの読み取り
                data = self.ser.read(length)
                
                # チェックサムの検証
                received_checksum = self.ser.read(1)[0]
                calculated_checksum = sum(data) & 0xFF
                if received_checksum != calculated_checksum:
                    print("チェックサムエラー")
                    return

                # フッターの確認
                footer = self.ser.read(1)[0]
                if footer != 0x55:
                    print("無効なフッター")
                    return

                # データの解析
                roll, pitch, yaw = struct.unpack('fff', data)
                print(f"受信データ: roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")
                # 回転行列の計算
                R = self.euler_to_rotation_matrix(roll, pitch, yaw)
                
                # 矢印の更新
                self.draw_arrows(R)
            
        except struct.error as e:
            print(f"データ解析エラー: {e}")
        except serial.SerialException as e:
            print(f"シリアル通信エラー: {e}")
        except Exception as e:
            print(f"その他のエラー: {e}")

    def start(self):
        """アニメーションの開始"""
        try:
            # アニメーションを開始
            self.ani = animation.FuncAnimation(self.fig, self.update_plot,
                                            interval=50, cache_frame_data=False)
            # メインループ
            plt.show()
            
        except KeyboardInterrupt:
            print("KeyboardInterrupt を検出")
        except Exception as e:
            print(f"予期せぬエラー: {e}")
        finally:
            try:
                if hasattr(self, 'ani'):
                    self.ani.event_source.stop()  # アニメーションを停止
                plt.close(self.fig)  # プロットウィンドウを閉じる
            except Exception:
                pass
            
            if hasattr(self, 'ser') and self.ser.is_open:
                self.ser.close()
                print("シリアルポートを閉じました")

if __name__ == "__main__":
    plotter = PosturePlotter()
    plotter.start()
