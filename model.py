import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ACCEL_SCALE = 2000
GYRO_SCALE = 300000

# Load the data
path = 'data/run_labels.csv'
df = pd.read_csv(path)
# カラム名を上書き
columns = ['accelX1', 'accelY1', 'accelZ1', 'gyroX1', 'gyroY1', 'gyroZ1', 'accelX2', 'accelY2', 'accelZ2', 'gyroX2', 'gyroY2', 'gyroZ2', 'label']
df.columns = columns

# 新たな特徴量を作る
accel_norm = (np.sqrt(df['accelX1']**2 + df['accelY1']**2 + df['accelZ1']**2) + np.sqrt(df['accelX2']**2 + df['accelY2']**2 + df['accelZ2']**2)) / 2
gyro_norm = (np.sqrt(df['gyroX1']**2 + df['gyroY1']**2 + df['gyroZ1']**2) + np.sqrt(df['gyroX2']**2 + df['gyroY2']**2 + df['gyroZ2']**2)) / 2
# 定数でスケールを1前後にする
df['accel_norm'] = accel_norm / ACCEL_SCALE
df['gyro_norm'] = gyro_norm / GYRO_SCALE