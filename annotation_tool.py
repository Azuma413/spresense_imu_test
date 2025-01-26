import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk
import csv

class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("IMU Data Annotation Tool")

        # Initialize variables
        self.imu_data = None
        self.video_frames = None
        self.labels = {}  # {(start_index, end_index): label}
        self.data_length = 0
        self.time_bar_value = tk.DoubleVar(value=0)
        self.current_frame_index = 0
        self.label_start_index = None

        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_widgets()
        
        # Add file selection button
        self.file_button = ttk.Button(self.root, text="Select Data File", command=self.select_file)
        self.file_button.pack(pady=5)

    def create_widgets(self):
        # Create top frame for image and plots
        self.top_frame = ttk.Frame(self.main_container)
        self.top_frame.pack(fill=tk.BOTH, expand=True)

        # Image Frame (上部)
        self.image_frame = ttk.Frame(self.top_frame)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 5))

        # 画像を中央に配置するためのフレーム
        self.image_center_frame = ttk.Frame(self.image_frame)
        self.image_center_frame.pack(expand=True)

        self.image_label = ttk.Label(self.image_center_frame)
        self.image_label.pack(anchor=tk.CENTER)
        self.display_placeholder_image()

        # Plot Frame (下部)
        self.plot_frame = ttk.Frame(self.top_frame)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(5, 0))

        # プロットのサイズを大きくして見やすくする
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.time_indicator_line = None

        # Bottom frame for controls
        self.bottom_frame = ttk.Frame(self.main_container)
        self.bottom_frame.pack(fill=tk.X, pady=(10, 0))

        # Time Bar
        self.time_bar = ttk.Scale(self.bottom_frame, orient=tk.HORIZONTAL,
                                from_=0, to=100,  # Will be updated when data is loaded
                                variable=self.time_bar_value)
        self.time_bar.pack(fill=tk.X, pady=(0, 10))
        self.time_bar.bind("<ButtonRelease-1>", self.update_time_bar)

        # Label Frame
        self.label_frame = ttk.Frame(self.root)
        self.label_frame.pack(pady=10)

        # Label input
        ttk.Label(self.label_frame, text="Label:").pack(side=tk.LEFT)
        self.label_entry = ttk.Entry(self.label_frame)
        self.label_entry.pack(side=tk.LEFT, padx=5)

        # Start/End label buttons
        self.start_label_button = ttk.Button(self.label_frame, text="Start Label", command=self.start_label)
        self.start_label_button.pack(side=tk.LEFT, padx=2)
        
        self.end_label_button = ttk.Button(self.label_frame, text="End Label", command=self.end_label)
        self.end_label_button.pack(side=tk.LEFT, padx=2)

        # Save Button
        ttk.Button(self.root, text="Save Labels", command=self.save_labels).pack(pady=10)

    def select_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="./data",
            title="Select Data File",
            filetypes=(("NumPy files", "*.npy"), ("All files", "*.*"))
        )
        if filepath:
            self.load_data(filepath)

    def load_data(self, filepath):
        try:
            data_dict = np.load(filepath, allow_pickle=True).item()
            self.imu_data = data_dict['imu_data']
            self.video_frames = data_dict.get('video_frames', None)
            
            if self.imu_data is not None:
                print(f"Loaded IMU data with shape: {self.imu_data.shape}")
                self.data_length = self.imu_data.shape[1]  # 時間次元のサイズ
                # Update time bar range
                self.time_bar.configure(from_=0, to=self.data_length-1)
                self.update_plot()
                
            if self.video_frames is None or len(self.video_frames) == 0:
                print("Warning: No video frames found in data file")
                self.video_frames = [Image.new('RGB', (400, 300), color='gray')]
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.imu_data = np.random.rand(12, 100)  # 12軸 x 100時点のダミーデータ
            self.video_frames = [Image.new('RGB', (400, 300), color='gray')]
            self.data_length = 100

    def display_placeholder_image(self):
        placeholder_image = Image.new('RGB', (400, 300), 'gray')
        placeholder_photo = ImageTk.PhotoImage(placeholder_image)
        self.image_label.config(image=placeholder_photo)
        self.image_label.image = placeholder_photo

    def resize_image(self, image):
        target_width = 400
        target_height = 300
        
        width, height = image.size
        aspect = width / height
        target_aspect = target_width / target_height
        
        if aspect > target_aspect:
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            new_height = target_height
            new_width = int(target_height * aspect)
            
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def update_plot(self):
        if self.imu_data is not None:
            self.ax.clear()
            time = np.arange(self.imu_data.shape[1])  # 時間次元のサイズ
            
            # IMUデータの1軸（加速度X）のみを表示
            self.ax.plot(time, self.imu_data[0, :], label='Acc X', linewidth=2, color='blue')
            
            self.ax.set_xlabel("Time [sample]")
            self.ax.set_ylabel("Sensor Values")
            self.ax.grid(True, alpha=0.3)
            
            # Y軸の表示範囲を適切に設定（1軸分のみ）
            y_min = np.min(self.imu_data[0, :])
            y_max = np.max(self.imu_data[0, :])
            margin = (y_max - y_min) * 0.1  # 10%のマージン
            self.ax.set_ylim(y_min - margin, y_max + margin)

            # 異なるラベルに対して異なる色を使用
            unique_labels = list(set(self.labels.values()))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            label_color_map = dict(zip(unique_labels, colors))
            
            # Draw labels with different colors
            for (start, end), label in self.labels.items():
                color = label_color_map[label]
                self.ax.axvspan(start, end, alpha=0.2, color=color, label=f'Label: {label}')

            if self.time_indicator_line:
                self.time_indicator_line.remove()
            current_time_index = int(self.time_bar_value.get())
            self.time_indicator_line = self.ax.axvline(x=current_time_index, color='r', linestyle='--')

            # Show start index marker if exists
            if self.label_start_index is not None:
                self.ax.axvline(x=self.label_start_index, color='g', linestyle='--', label='Label Start')

            # 重複するラベルを統合して凡例を表示（位置を左に変更）
            handles, labels = self.ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            self.ax.legend(by_label.values(), by_label.keys(), 
                         loc='center left', bbox_to_anchor=(1.02, 0.5),
                         borderaxespad=0)
            
            # グラフのレイアウトを調整して凡例が切れないようにする
            self.fig.tight_layout()

            self.canvas.draw()

    def update_time_bar(self, event=None):
        if self.imu_data is None:
            return
            
        current_index = int(self.time_bar_value.get())
        self.current_frame_index = current_index
        self.update_plot()
        self.update_image_frame(current_index)

    def update_image_frame(self, frame_index):
        try:
            if self.video_frames is not None and len(self.video_frames) > 0:
                frame_index = min(frame_index, len(self.video_frames) - 1)
                frame_index = max(frame_index, 0)
                video_frame = self.video_frames[frame_index]
                
                if isinstance(video_frame, np.ndarray):
                    if video_frame.dtype == np.uint8:
                        # RGB順序を修正
                        if len(video_frame.shape) == 3 and video_frame.shape[2] == 3:
                            video_frame = video_frame[..., ::-1]  # BGRからRGBに変換
                        video_frame = Image.fromarray(video_frame)
                    else:
                        video_frame = ((video_frame - video_frame.min()) * (255.0 / (video_frame.max() - video_frame.min()))).astype(np.uint8)
                        video_frame = Image.fromarray(video_frame)
                elif not isinstance(video_frame, Image.Image):
                    raise ValueError(f"Unsupported image type: {type(video_frame)}")
                
                video_frame = self.resize_image(video_frame)
                photo = ImageTk.PhotoImage(video_frame)
                self.image_label.config(image=photo)
                self.image_label.image = photo
            else:
                self.display_placeholder_image()
        except Exception as e:
            print(f"Error updating image frame: {e}")
            self.display_placeholder_image()

    def start_label(self):
        self.label_start_index = int(self.time_bar_value.get())
        print(f"Label start set at index {self.label_start_index}")
        self.update_plot()

    def end_label(self):
        if self.label_start_index is None:
            print("Please set start index first")
            return

        end_index = int(self.time_bar_value.get())
        label_text = self.label_entry.get()
        
        if not label_text:
            print("Please enter a label")
            return

        # Ensure start_index is less than end_index
        start_index = min(self.label_start_index, end_index)
        end_index = max(self.label_start_index, end_index)
        
        self.labels[(start_index, end_index)] = label_text
        print(f"Label '{label_text}' set from index {start_index} to {end_index}")
        
        self.label_start_index = None
        self.update_plot()

    def save_labels(self):
        if self.imu_data is None:
            print("No IMU data to save")
            return
            
        # ラベル情報の保存
        label_filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="imu_labels.csv"
        )
        
        if not label_filepath:
            return
            
        with open(label_filepath, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # ヘッダー行の書き込み
            header = [f'IMU_{i}' for i in range(12)]  # 12軸のIMUデータ
            header.append('Label')  # ラベル列
            csv_writer.writerow(header)
            
            # 各時間ステップのデータとラベルを保存
            for t in range(self.data_length):
                # IMUデータの取得
                imu_values = self.imu_data[:, t].tolist()
                
                # 現在の時間ステップのラベルを探す
                current_label = "None"  # デフォルトラベル
                for (start_index, end_index), label in self.labels.items():
                    if start_index <= t <= end_index:
                        current_label = label
                        break
                
                # IMUデータとラベルを結合して書き出し
                row = imu_values + [current_label]
                csv_writer.writerow(row)
                
        print(f"IMU data and labels saved to {label_filepath}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()
