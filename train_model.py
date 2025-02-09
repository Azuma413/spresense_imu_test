import os
import torch
from torch.utils.data import DataLoader
from model import IMUDataset, IMUPredictor, IMULinearRegression
import numpy as np
from tqdm import tqdm

def train(model, train_loader, valid_loader, num_epochs=50, device='cuda', save_path='models/best_model.pth'):
    # Create directory for saved models
    os.makedirs('models', exist_ok=True)
    # Initialize training components
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    print(f"\nStarting training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print("-" * 60)
    best_acc = 0
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        # Validation
        model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for features, labels in tqdm(valid_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Valid]'):
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()
        valid_loss = valid_loss / len(valid_loader)
        valid_acc = 100. * valid_correct / valid_total
        # Save best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            model.save(save_path)
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%')
        print('-' * 60)

def main():
    # model_name = "transformer"
    model_name = "linear"
    try:
        # Parameters
        WINDOW_SIZE = 60
        BATCH_SIZE = 32
        NUM_CLASSES = 4  # run, walk, shake, something
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Check CUDA availability
        if not torch.cuda.is_available() and DEVICE == 'cuda':
            print("CUDA is not available. Using CPU instead.")
            DEVICE = 'cpu'
        # データパス（CSVファイルのみ使用）
        data_paths = [
            'data/run_labels.csv',
            'data/walk_labels.csv',
            'data/shake_labels.csv',
            'data/something_labels.csv'
        ]
        # Verify all data files exist
        for path in data_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")
        print("\nInitializing datasets...")
        # Dataset and DataLoader
        train_dataset = IMUDataset(data_paths, window_size=WINDOW_SIZE, mode='train')
        valid_dataset = IMUDataset(data_paths, window_size=WINDOW_SIZE, mode='valid')
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(valid_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
        # Model
        print("\nInitializing model...")
        if model_name == "transformer":
            model = IMUPredictor(num_classes=NUM_CLASSES).to(DEVICE)
        else:
            model = IMULinearRegression(num_classes=NUM_CLASSES, window_size=WINDOW_SIZE).to(DEVICE)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        # Training
        train(model, train_loader, valid_loader, device=DEVICE, save_path=f'models/best_{model_name}_model.pth')
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
