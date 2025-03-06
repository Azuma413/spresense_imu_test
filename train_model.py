import os
import torch
from torch.utils.data import DataLoader
from model import IMUDataset, IMUPredictor, IMULinearRegression, IMUConvNet
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(model, train_loader, valid_loader, num_epochs=50, device='cuda', save_path='models/best_model.pth'):
    # Create directories for saved models and plots
    os.makedirs('models', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    
    # Lists to store accuracy values
    train_accuracies = []
    valid_accuracies = []
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
        
        # Store accuracies
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)
    
    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    model_type = save_path.split('_')[-2]  # Extract model type from save path
    plt.savefig(f'images/accuracy_plot_{model_type}.png')
    plt.close()

def main(model_name):
    try:
        # Parameters
        WINDOW_SIZE = 30
        BATCH_SIZE = 16 # 32
        FEATURE_DIM = 2*4
        NUM_CLASSES = 3  # run, walk, something
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # データパス（CSVファイルのみ使用）
        train_data_paths = [
            'data/train1_labels.csv',
            'data/train2_labels.csv',
            'data/train3_labels.csv',
        ]
        eval_data_paths = [
            'data/eval_labels.csv',
        ]
        print("\nInitializing datasets...")
        # Dataset and DataLoader
        train_dataset = IMUDataset(train_data_paths, window_size=WINDOW_SIZE)
        valid_dataset = IMUDataset(eval_data_paths, window_size=WINDOW_SIZE)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(valid_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
        # Model
        print("\nInitializing model...")
        if model_name == "transformer":
            model = IMUPredictor(num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM, embed_dim=128, num_heads=8, num_layers=4).to(DEVICE)
        elif model_name == "conv":
            model = IMUConvNet(num_classes=NUM_CLASSES, window_size=WINDOW_SIZE, feature_dim=FEATURE_DIM).to(DEVICE)
        elif model_name == "linear":
            model = IMULinearRegression(num_classes=NUM_CLASSES, window_size=WINDOW_SIZE, num_features=FEATURE_DIM).to(DEVICE)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        # Training
        train(model, train_loader, valid_loader, num_epochs=10, device=DEVICE, save_path=f'models/best_{model_name}_model.pth')
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == '__main__':
    model_name = "transformer"
    # model_name = "linear"
    # model_name = "conv"
    try:
        main(model_name)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
