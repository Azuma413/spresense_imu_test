import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np
import datetime

ACCEL_SCALE = 2000
GYRO_SCALE = 300000

class IMUDataset(Dataset):
    def __init__(self, data_paths, window_size=60, mode="train"):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        
        # Label mapping
        self.label_to_idx = {
            'run': 0,
            'walk': 1,
            'shake': 2,
            'something': 3
        }
        
        # データの読み込みと前処理
        dfs = []
        for path in data_paths:
            try:
                df = pd.read_csv(path)
                # カラム名を統一
                columns = ['accelX1', 'accelY1', 'accelZ1', 'gyroX1', 'gyroY1', 'gyroZ1', 
                          'accelX2', 'accelY2', 'accelZ2', 'gyroX2', 'gyroY2', 'gyroZ2', 'label']
                df.columns = columns
                
                # データ分割（各ファイルごとに分割）
                split_idx = len(df) - int(len(df)*0.1)
                if mode == "train":
                    df = df.iloc[:split_idx]
                elif mode == "valid":
                    df = df.iloc[split_idx:]
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")
                continue

            # 特徴量の作成
            try:
                accel_norm = (np.sqrt(df['accelX1']**2 + df['accelY1']**2 + df['accelZ1']**2) + 
                            np.sqrt(df['accelX2']**2 + df['accelY2']**2 + df['accelZ2']**2)) / 2
                gyro_norm = (np.sqrt(df['gyroX1']**2 + df['gyroY1']**2 + df['gyroZ1']**2) + 
                            np.sqrt(df['gyroX2']**2 + df['gyroY2']**2 + df['gyroZ2']**2)) / 2
                
                df['accel_norm'] = accel_norm / ACCEL_SCALE
                df['gyro_norm'] = gyro_norm / GYRO_SCALE
                
                # Convert string labels to integers
                df['label'] = df['label'].map(self.label_to_idx)
                
                dfs.append(df)
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                continue
        
        # 前処理済みデータの結合
        self.df = pd.concat(dfs, ignore_index=True)
            
    def __len__(self):
        return len(self.df) - self.window_size

    def __getitem__(self, idx):
        window = self.df.iloc[idx:idx+self.window_size]
        
        # 特徴量の抽出（2個のノルム）
        features = window[['accel_norm', 'gyro_norm']].values.astype(np.float32)
        
        # ウィンドウ内で最も頻出するラベルを教師データとする
        label = window['label'].mode()[0]
        
        return torch.FloatTensor(features), torch.LongTensor([label])[0]

class IMUPredictor(torch.nn.Module):
    def __init__(self, num_classes, feature_dim=2, embed_dim=64, num_heads=4, num_layers=2):
        super(IMUPredictor, self).__init__()
        
        self.feature_tokenizer = torch.nn.Linear(feature_dim, embed_dim)
        self.batch_norm1 = torch.nn.BatchNorm1d(embed_dim)
        
        self.predictor = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim*4,
                dropout=0.1,
                activation=torch.nn.SiLU(),
            ),
            num_layers=num_layers,
        )
        
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.batch_norm2 = torch.nn.BatchNorm1d(embed_dim)
        self.output_layer = torch.nn.Linear(embed_dim, num_classes)
        self.act = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        """
        x: (batch_size, seq_len, feature_dim)
        return: (batch_size, num_classes)
        """
        x = self.feature_tokenizer(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = self.batch_norm1(x)
        x = x.permute(2, 0, 1)  # (seq_len, batch, embed_dim)

        x = self.predictor(x)  # (seq_len, batch, embed_dim)
        x = x.permute(1, 2, 0)  # (batch, embed_dim, seq_len)
        x = self.pool(x)  # (batch, embed_dim, 1)
        x = x.squeeze(-1)  # (batch, embed_dim)

        x = self.batch_norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.output_layer(x)  # (batch, num_classes)
        return x

    def save(self, path):
        print(f"save model to {path}")
        torch.save(self.state_dict(), path)

    def load(self, path):
        print(f"load model from {path}")
        self.load_state_dict(torch.load(path))
