# IMU動作認識システム

このプロジェクトは、Spresense向けのIMU（慣性計測装置）センサを用いた動作認識システムです。2つのIMUセンサからのデータを収集し、機械学習モデルを用いてリアルタイムで動作を分類します。

## 機能

- IMUセンサ（ISM330DHCX）からの加速度・ジャイロデータの取得
- Wi-Fi経由でのセンサデータのリアルタイム転送
- データの可視化とアノテーション
- 機械学習モデルの学習
- リアルタイムでの動作認識

## フォルダ構成

```
.
├── data/                     # 学習用データセット
│   ├── run_labels.csv        # 走る動作のラベル付きデータ
│   ├── walk_labels.csv       # 歩く動作のラベル付きデータ
│   ├── shake_labels.csv      # 振る動作のラベル付きデータ
│   └── something_labels.csv  # その他の動作のラベル付きデータ
├── imu_test/                 # Spresense用ソースコード
│   ├── imu_test.ino          # メインのArduinoスケッチ
│   └── udp_send.hpp          # UDPデータ送信用ライブラリ
├── models/                   # 学習済みモデル
│   └── best_model.pth        # 最良の学習結果を保存したモデル
└── *.py                      # Pythonスクリプト群
```

## 必要な環境

### Spresense側
- Arduino IDE
- Spresense Arduino互換ライブラリ
- ISM330DHCXライブラリ（SparkFun_ISM330DHCX）

### PC側
- Python 3.x
- PyTorch
- NumPy
- pandas
- matplotlib
- OpenCV (cv2)

## セットアップと実行方法

### 1. Spresenseのセットアップ
1. Arduino IDEでimu_test/imu_test.inoを開く
2. Wi-Fi設定を環境に合わせて修正
   ```cpp
   const char* ssid = "your_wifi_ssid";
   const char* password = "your_wifi_password";
   ```
3. Spresenseにスケッチを書き込む

### 2. データ収集と学習
1. データ収集モードの実行
   ```bash
   python plotter.py
   ```
   - リアルタイムでIMUデータと画像が表示される
   - 終了時（Ctrl+C）に自動でデータが保存される

2. データのアノテーション
   ```bash
   python annotation_tool.py
   ```
   - GUIツールが起動
   - 「Select Data File」で.npyファイルを選択
   - 波形を見ながら動作区間を選択してラベル付け
   - 「Save Labels」でラベル付きデータを保存

3. モデルの学習
   ```bash
   python train_model.py
   ```
   - data/フォルダ内のラベル付きデータを使用して学習
   - 学習済みモデルはmodels/best_model.pthに保存

### 3. リアルタイム推論
```bash
python realtime_inference.py
```
- カメラ映像とIMUデータをリアルタイムで表示
- 学習済みモデルを使用して動作を認識・表示

## 実装の詳細

### IMUデータの取得（imu_test.ino）
- 2つのISM330DHCXセンサから加速度とジャイロデータを取得
- センサの設定：
  - 加速度：208Hz、±4g、LPフィルタ適用
  - ジャイロ：104Hz、±250dps、LPフィルタ適用
- 10サンプルの移動平均を計算
- UDPでデータを送信

### モデルアーキテクチャ（model.py）
- Transformerベースのアーキテクチャを採用
- 入力：
  - 時系列長：60サンプル
  - 特徴量：加速度ノルム、ジャイロノルム（2次元）
- 出力：4クラス分類（run, walk, shake, something）

### データの前処理
- 加速度・ジャイロデータのノルム計算
- スケーリング（ACCEL_SCALE=2000, GYRO_SCALE=300000）
- 60サンプルの時系列ウィンドウでバッチ化
