import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Mengimpor fungsi dan class dari file lain yang sudah dibuat
from utils.preprocessing import preprocess_csi_signal
from utils.dataset_loader import get_data_loaders
from models.baseline_cnn import Baseline1DCNN
from models.proposed_resnet import ProposedCSIResNet

def train_and_evaluate(model, train_loader, val_loader, epochs=20, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model, history

def plot_training_history(history_baseline, history_proposed):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_baseline['train_loss'], label='Baseline Train Loss', linestyle='--')
    plt.plot(history_baseline['val_loss'], label='Baseline Val Loss', linestyle='-')
    plt.plot(history_proposed['train_loss'], label='Proposed Train Loss', linestyle='--')
    plt.plot(history_proposed['val_loss'], label='Proposed Val Loss', linestyle='-')
    plt.title('Perbandingan Loss Model (Baseline vs Proposed)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history_baseline['train_acc'], label='Baseline Train Acc', linestyle='--')
    plt.plot(history_baseline['val_acc'], label='Baseline Val Acc', linestyle='-')
    plt.plot(history_proposed['train_acc'], label='Proposed Train Acc', linestyle='--')
    plt.plot(history_proposed['val_acc'], label='Proposed Val Acc', linestyle='-')
    plt.title('Perbandingan Akurasi Model (Baseline vs Proposed)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_and_plot_confusion_matrix(model, val_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Model Evaluasi')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    print("Memuat dataset UT-HAR dari direktori lokal Colab...")
    data_dir = '/content/UT-HAR-Dataset/data/'
    label_dir = '/content/UT-HAR-Dataset/label/'

    X_train_raw = np.load(data_dir + 'X_train.csv', allow_pickle=True).astype(np.float32)
    X_val_raw = np.load(data_dir + 'X_val.csv', allow_pickle=True).astype(np.float32)
    y_train = np.load(label_dir + 'y_train.csv', allow_pickle=True).squeeze().astype(int)
    y_val = np.load(label_dir + 'y_val.csv', allow_pickle=True).squeeze().astype(int)

    if len(X_train_raw.shape) == 2:
        print("Data terdeteksi sebagai 2D. Melakukan reshape ke 3D...")
        total_fitur = X_train_raw.shape[1]
        in_channels = 10
        time_steps = total_fitur // in_channels
        X_train = X_train_raw.reshape(-1, in_channels, time_steps)
        X_val = X_val_raw.reshape(-1, in_channels, time_steps)
    else:
        print("Data sudah berbentuk 3D. Melewati proses reshape...")
        X_train = X_train_raw
        X_val = X_val_raw
        in_channels = X_train.shape[1]
        time_steps = X_train.shape[2]

    num_classes = len(np.unique(y_train))

    print("\nMenjalankan preprocessing (Low-pass filter & Z-score normalization)...")
    X_train_processed = preprocess_csi_signal(X_train)
    X_val_processed = preprocess_csi_signal(X_val)

    train_loader, val_loader = get_data_loaders(X_train_processed, y_train, X_val_processed, y_val, batch_size=32)

    print("\n--- Training Model Baseline (1D-CNN) ---")
    baseline_model = Baseline1DCNN(in_channels=in_channels, num_classes=num_classes)
    trained_baseline, history_baseline = train_and_evaluate(baseline_model, train_loader, val_loader, epochs=10, lr=0.001)

    print("\n--- Training Model Usulan (Proposed ResNet) ---")
    proposed_model = ProposedCSIResNet(in_channels=in_channels, num_classes=num_classes)
    trained_proposed, history_proposed = train_and_evaluate(proposed_model, train_loader, val_loader, epochs=10, lr=0.001)

    print("\nMenampilkan grafik perbandingan performa model...")
    plot_training_history(history_baseline, history_proposed)

    class_names = ['Berbaring', 'Jatuh', 'Berjalan', 'Memungut', 'Berlari', 'Duduk', 'Berdiri']
    print("\nEvaluasi Akhir Model Usulan (Proposed ResNet):")
    evaluate_and_plot_confusion_matrix(trained_proposed, val_loader, class_names)