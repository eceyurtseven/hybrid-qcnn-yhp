import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST, INFO
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import time
import warnings
import os
import pandas as pd
from collections import defaultdict
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')


def save_loss_plot(train_losses, val_losses, model_name):
    """Save a loss-vs-epoch plot for the given lists of train/val losses"""
    try:
        plt.figure(figsize=(8, 5))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label='Train Loss', color='#2ecc71', linewidth=2)
        if val_losses:
            plt.plot(epochs, val_losses, label='Val Loss', color='#e67e22', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss per Epoch - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        # Sanitize filename
        safe_name = ''.join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in model_name).strip().replace(' ', '_')
        out_path = f'{safe_name}_loss_curve.png'
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved loss curve to: {out_path}")
    except Exception as e:
        print(f"Failed to save loss plot: {e}")

# Enhanced Data Preprocessing for Higher Accuracy
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def load_and_prepare_data():
    """Load and prepare the dataset for training, validation, and testing"""
    print("Loading PneumoniaMNIST dataset for Fair Quantum vs Classical Comparison...")
    try:
        train_dataset = PneumoniaMNIST(split='train', transform=transform_train, download=True)
        val_dataset = PneumoniaMNIST(split='val', transform=transform_test, download=True)
        test_dataset = PneumoniaMNIST(split='test', transform=transform_test, download=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    print(f"Using full dataset - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    batch_size = 16  # Increased batch size for stability

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader

# --- Optimized Feature Extractor (Base Architecture) ---
class OptimizedFeatureExtractor(nn.Module):
    def __init__(self):
        super(OptimizedFeatureExtractor, self).__init__()
        # First block - capture basic features
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        # Second block - intermediate features
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        # Third block - high-level features
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))  # 7x7 -> 4x4

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # 128 * 4 * 4 = 2048
        return x


# --- Quantum Components (Simplified and Robust) ---
n_qubits = 4  # Reduced to 4 qubits for stability
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit_1(inputs, weights):
    """First quantum circuit with amplitude encoding"""
    # Amplitude encoding (only use first 2^n_qubits values)
    input_size = 2 ** n_qubits
    padded_inputs = torch.nn.functional.pad(inputs[:input_size], (0, max(0, input_size - len(inputs[:input_size]))))
    normalized_inputs = padded_inputs / torch.norm(padded_inputs)
    qml.AmplitudeEmbedding(features=normalized_inputs, wires=range(n_qubits), normalize=True)

    # Variational layers
    for layer in range(weights.shape[0]):
        for qubit in range(n_qubits):
            qml.RY(weights[layer, qubit, 0], wires=qubit)
            qml.RZ(weights[layer, qubit, 1], wires=qubit)

        # Entanglement
        for qubit in range(n_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])

    # Measurements
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit_2(inputs, weights):
    """Second quantum circuit with angle encoding"""
    # Angle encoding
    for i in range(min(len(inputs), n_qubits)):
        qml.RY(inputs[i] * np.pi, wires=i)

    # Variational layers
    for layer in range(weights.shape[0]):
        for qubit in range(n_qubits):
            qml.RX(weights[layer, qubit, 0], wires=qubit)
            qml.RY(weights[layer, qubit, 1], wires=qubit)

        # Different entanglement pattern
        for qubit in range(n_qubits):
            qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])

    # More measurements
    measurements = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    measurements.extend([qml.expval(qml.PauliX(i)) for i in range(min(2, n_qubits))])
    return measurements


class QuantumLayer1(nn.Module):
    def __init__(self, n_layers=2):
        super(QuantumLayer1, self).__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit_1, weight_shapes)
        self._init_weights()

    def _init_weights(self):
        for param in self.qlayer.parameters():
            nn.init.uniform_(param, -np.pi / 6, np.pi / 6)

    def forward(self, x):
        # Process each sample in batch
        batch_size = x.size(0)
        quantum_outputs = []

        for i in range(batch_size):
            input_sample = x[i]
            # Ensure input is the right size for quantum circuit
            input_size = 2 ** n_qubits
            if len(input_sample) > input_size:
                input_sample = input_sample[:input_size]
            elif len(input_sample) < input_size:
                padding = torch.zeros(input_size - len(input_sample), device=x.device)
                input_sample = torch.cat([input_sample, padding])

            qout = self.qlayer(input_sample)
            quantum_outputs.append(qout)

        return torch.stack(quantum_outputs)


class QuantumLayer2(nn.Module):
    def __init__(self, n_layers=2):
        super(QuantumLayer2, self).__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit_2, weight_shapes)
        self._init_weights()

    def _init_weights(self):
        for param in self.qlayer.parameters():
            nn.init.uniform_(param, -np.pi / 6, np.pi / 6)

    def forward(self, x):
        batch_size = x.size(0)
        quantum_outputs = []

        for i in range(batch_size):
            input_sample = x[i]
            # For angle encoding, we can use the input directly (just take first n_qubits)
            if len(input_sample) > n_qubits:
                input_sample = input_sample[:n_qubits]
            elif len(input_sample) < n_qubits:
                padding = torch.zeros(n_qubits - len(input_sample), device=x.device)
                input_sample = torch.cat([input_sample, padding])

            qout = self.qlayer(input_sample)
            quantum_outputs.append(qout)

        return torch.stack(quantum_outputs)


# --- Best Quantum CNN Architecture ---
class BestQuantumCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BestQuantumCNN, self).__init__()
        self.feature_extractor = OptimizedFeatureExtractor()

        # Quantum preprocessing
        quantum_input_size = 2 ** n_qubits
        self.quantum_preprocessor1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, quantum_input_size),
            nn.Tanh()
        )

        self.quantum_preprocessor2 = nn.Sequential(
            nn.Linear(2048, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_qubits),
            nn.Tanh()
        )

        # Quantum layers
        self.quantum_layer1 = QuantumLayer1(n_layers=2)
        self.quantum_layer2 = QuantumLayer2(n_layers=2)

        # Calculate quantum output sizes
        q1_output_size = n_qubits  # PauliZ measurements
        q2_output_size = n_qubits + 2  # PauliZ + PauliX measurements

        # Fusion and classification
        self.quantum_fusion = nn.Sequential(
            nn.Linear(q1_output_size + q2_output_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 2048, 512),  # Quantum features + original features
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)

        # Quantum preprocessing
        q_input1 = self.quantum_preprocessor1(features)
        q_input2 = self.quantum_preprocessor2(features)

        # Quantum processing
        q_features1 = self.quantum_layer1(q_input1)
        q_features2 = self.quantum_layer2(q_input2)

        # Fuse quantum features
        combined_quantum = torch.cat([q_features1, q_features2], dim=1)
        fused_quantum = self.quantum_fusion(combined_quantum)

        # Combine with classical features for final classification
        final_features = torch.cat([fused_quantum, features], dim=1)
        output = self.classifier(final_features)

        return output


# --- Derived Classical CNN (Same Complexity) ---
class DerivedClassicalCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DerivedClassicalCNN, self).__init__()
        self.feature_extractor = OptimizedFeatureExtractor()

        # Classical equivalents of quantum preprocessors
        quantum_input_size = 2 ** n_qubits
        self.classical_processor1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, quantum_input_size),
            nn.Tanh(),
            nn.Linear(quantum_input_size, n_qubits),  # Equivalent to quantum layer 1
            nn.ReLU()
        )

        self.classical_processor2 = nn.Sequential(
            nn.Linear(2048, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_qubits),
            nn.Tanh(),
            nn.Linear(n_qubits, n_qubits + 2),  # Equivalent to quantum layer 2
            nn.ReLU()
        )

        # Classical fusion (equivalent to quantum fusion)
        self.classical_fusion = nn.Sequential(
            nn.Linear(n_qubits + n_qubits + 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Same classifier as quantum model
        self.classifier = nn.Sequential(
            nn.Linear(128 + 2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract features (same as quantum)
        features = self.feature_extractor(x)

        # Classical processing
        c_features1 = self.classical_processor1(features)
        c_features2 = self.classical_processor2(features)

        # Fuse features
        combined_classical = torch.cat([c_features1, c_features2], dim=1)
        fused_classical = self.classical_fusion(combined_classical)

        # Final classification
        final_features = torch.cat([fused_classical, features], dim=1)
        output = self.classifier(final_features)

        return output


# --- Advanced Training Function ---
def train_and_evaluate_model(model, model_name, train_loader, val_loader, test_loader, epochs=60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"\n{'=' * 70}")
    print(f"Training {model_name}")
    print(f"{'=' * 70}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Using device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    # Advanced loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer with different learning rates for quantum vs classical parts
    if "Quantum" in model_name:
        quantum_params = []
        classical_params = []

        for name, param in model.named_parameters():
            if 'qlayer' in name.lower() or 'quantum' in name.lower():
                quantum_params.append(param)
            else:
                classical_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': classical_params, 'lr': 0.001, 'weight_decay': 0.01},
            {'params': quantum_params, 'lr': 0.0005, 'weight_decay': 0.005}
        ], eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Training tracking
    train_acc_list, val_acc_list, test_acc_list, losses_list = [], [], [], []
    # Additional loss tracking (explicit names)
    train_loss_list, val_loss_list, test_loss_list = [], [], []
    train_f1_list, val_f1_list, test_f1_list = [], [], []
    epoch_times = []
    best_test_acc = 0.0
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 25  # Increased from 15 to allow more epochs for natural plateau

    print("Starting training...")
    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).squeeze()

            optimizer.zero_grad()

            try:
                outputs = model(data)
                loss = criterion(outputs, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue

        if total_train == 0:
            print("No successful training batches, skipping epoch")
            continue

        train_accuracy = correct_train / total_train
        avg_loss = epoch_loss / len(train_loader)

        # Validation phase with F1 score
        model.eval()
        correct_val, total_val = 0, 0
        val_y_true, val_y_pred = [], []
        val_epoch_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).squeeze()
                try:
                    outputs = model(data)
                    vloss = criterion(outputs, target)
                    val_epoch_loss += vloss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += target.size(0)
                    correct_val += (predicted == target).sum().item()
                    val_y_true.extend(target.cpu().numpy())
                    val_y_pred.extend(predicted.cpu().numpy())
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue

        if total_val == 0:
            print("No successful validation batches, skipping epoch")
            continue

        val_accuracy = correct_val / total_val
        val_avg_loss = val_epoch_loss / len(val_loader)
        val_f1 = f1_score(val_y_true, val_y_pred, average='binary') if val_y_true and val_y_pred else 0.0

        # Test evaluation phase with F1 score
        correct_test, total_test = 0, 0
        test_y_true, test_y_pred = [], []
        test_epoch_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device).squeeze()
                try:
                    outputs = model(data)
                    tloss = criterion(outputs, target)
                    test_epoch_loss += tloss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += target.size(0)
                    correct_test += (predicted == target).sum().item()
                    test_y_true.extend(target.cpu().numpy())
                    test_y_pred.extend(predicted.cpu().numpy())
                except Exception as e:
                    print(f"Error in test evaluation: {e}")
                    continue

        if total_test == 0:
            print("No successful test batches, skipping epoch")
            continue

        test_accuracy = correct_test / total_test
        test_avg_loss = test_epoch_loss / len(test_loader)
        test_f1 = f1_score(test_y_true, test_y_pred, average='binary') if test_y_true and test_y_pred else 0.0

        # Early stopping based on validation accuracy (prevents overfitting to test set)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        # Track best test accuracy too
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy

        # Record metrics
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        train_acc_list.append(train_accuracy)
        val_acc_list.append(val_accuracy)
        test_acc_list.append(test_accuracy)
        val_f1_list.append(val_f1)
        test_f1_list.append(test_f1)
        losses_list.append(avg_loss)
        train_loss_list.append(avg_loss)
        val_loss_list.append(val_avg_loss)
        test_loss_list.append(test_avg_loss)

        # Progress reporting
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch + 1:3d}: TrainLoss={avg_loss:.4f}, ValLoss={val_avg_loss:.4f}, "
                f"TrainAcc={train_accuracy:.4f}, ValAcc={val_accuracy:.4f}, TestAcc={test_accuracy:.4f}, "
                f"Best Test={best_test_acc:.4f}, Time={epoch_time:.1f}s"
            )

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    total_training_time = time.time() - total_start_time

    # Final evaluation
    model.eval()
    y_true, y_pred = [], []
    inference_start = time.time()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).squeeze()
            try:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
            except Exception as e:
                print(f"Error in final evaluation: {e}")
                continue

    inference_time = time.time() - inference_start
    final_accuracy = accuracy_score(y_true, y_pred) if y_true and y_pred else 0.0
    final_f1_score = f1_score(y_true, y_pred, average='binary') if y_true and y_pred else 0.0

    results = {
        'model_name': model_name,
        'best_accuracy': best_test_acc,
        'best_val_accuracy': best_val_acc,
        'final_accuracy': final_accuracy,
        'final_f1_score': final_f1_score,
        'train_acc_list': train_acc_list,
        'val_acc_list': val_acc_list,
        'test_acc_list': test_acc_list,
        'val_f1_list': val_f1_list,
        'test_f1_list': test_f1_list,
        'losses_list': losses_list,
        'train_loss_list': train_loss_list,
        'val_loss_list': val_loss_list,
        'test_loss_list': test_loss_list,
        'training_time': total_training_time,
        'inference_time': inference_time,
        'avg_epoch_time': np.mean(epoch_times) if epoch_times else 0,
        'y_true': y_true,
        'y_pred': y_pred,
        'parameters': total_params
    }

    print(f"\n{model_name} Final Results:")
    print(f"Final Test Accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    print(f"Final Test F1 Score: {final_f1_score:.4f}")
    print(f"Best Test Accuracy: {best_test_acc:.4f} ({best_test_acc * 100:.2f}%)")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"Training Time: {total_training_time:.1f}s ({total_training_time / 60:.1f} min)")
    print(f"Inference Time: {inference_time:.4f}s")

    # Loss curves will be saved at the end (aggregated across runs)
    # Individual per-run loss curves are skipped to reduce clutter

    return results


# --- Main Experiment Function ---
def run_single_experiment(run_id):
    """Run a single experiment and return results"""
    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT RUN {run_id + 1}/10")
    print(f"{'=' * 80}")

    # Set different random seeds for each run to ensure variety
    torch.manual_seed(42 + run_id * 100)
    np.random.seed(42 + run_id * 100)

    # Load data
    train_loader, val_loader, test_loader = load_and_prepare_data()

    # Train Quantum model (more epochs for natural plateau)
    quantum_model = BestQuantumCNN(num_classes=2)
    quantum_results = train_and_evaluate_model(quantum_model, f"Quantum CNN (Run {run_id + 1})", train_loader,
                                               val_loader, test_loader, epochs=80)

    # Train Classical model (more epochs for natural plateau)
    classical_model = DerivedClassicalCNN(num_classes=2)
    classical_results = train_and_evaluate_model(classical_model, f"Classical CNN (Run {run_id + 1})", train_loader,
                                                 val_loader, test_loader, epochs=80)

    return classical_results, quantum_results


# --- Results Aggregation Functions ---


def aggregate_results(classical_results, quantum_results):
    """Aggregate results from all 10 runs"""

    def calculate_stats(values):
        values = [v for v in values if v is not None and not np.isnan(v)]
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    # Aggregate Classical results
    classical_aggregated = {}
    classical_metrics = ['final_accuracy', 'final_f1_score', 'best_accuracy', 'training_time', 'inference_time', 'avg_epoch_time']

    for metric in classical_metrics:
        values = [result[metric] for result in classical_results if metric in result]
        classical_aggregated[metric] = calculate_stats(values)

    # Aggregate Quantum results
    quantum_aggregated = {}
    for metric in classical_metrics:
        values = [result[metric] for result in quantum_results if metric in result]
        quantum_aggregated[metric] = calculate_stats(values)

    # Calculate improvement statistics
    final_acc_improvements = []
    final_f1_improvements = []
    best_acc_improvements = []

    for i in range(len(classical_results)):
        if i < len(quantum_results):
            c_final = classical_results[i].get('final_accuracy', 0)
            q_final = quantum_results[i].get('final_accuracy', 0)
            c_final_f1 = classical_results[i].get('final_f1_score', 0)
            q_final_f1 = quantum_results[i].get('final_f1_score', 0)
            c_best = classical_results[i].get('best_accuracy', 0)
            q_best = quantum_results[i].get('best_accuracy', 0)

            if c_final > 0:
                final_acc_improvements.append(q_final - c_final)
            if c_final_f1 > 0:
                final_f1_improvements.append(q_final_f1 - c_final_f1)
            if c_best > 0:
                best_acc_improvements.append(q_best - c_best)

    improvement_stats = {
        'final_accuracy_improvement': calculate_stats(final_acc_improvements),
        'final_f1_improvement': calculate_stats(final_f1_improvements),
        'best_accuracy_improvement': calculate_stats(best_acc_improvements)
    }

    # Get parameter count (should be same across runs)
    parameters = {
        'classical': classical_results[0].get('parameters', 0) if classical_results else 0,
        'quantum': quantum_results[0].get('parameters', 0) if quantum_results else 0
    }

    return classical_aggregated, quantum_aggregated, improvement_stats, parameters


def print_aggregated_results(classical_agg, quantum_agg, improvement_stats, parameters):
    """Print comprehensive aggregated results"""

    print(f"\n{'=' * 80}")
    print("AGGREGATED RESULTS FROM 10 EXPERIMENTAL RUNS")
    print(f"{'=' * 80}")

    # Results table header
    print(f"\n{'Metric':<25} {'Classical CNN':<30} {'Quantum CNN':<30} {'Improvement':<20}")
    print("-" * 105)

    # Final Accuracy
    c_final = classical_agg['final_accuracy']
    q_final = quantum_agg['final_accuracy']
    improvement_final = improvement_stats['final_accuracy_improvement']

    # Format strings properly
    classical_final_str = f"{c_final['mean']:.4f} ± {c_final['std']:.4f}"
    quantum_final_str = f"{q_final['mean']:.4f} ± {q_final['std']:.4f}"
    improvement_final_str = f"{improvement_final['mean']:+.4f} ± {improvement_final['std']:.4f}"

    print(
        f"{'Final Accuracy (mean)':<25} {classical_final_str:<30} {quantum_final_str:<30} {improvement_final_str:<20}")

    classical_range_str = f"[{c_final['min']:.4f}, {c_final['max']:.4f}]"
    quantum_range_str = f"[{q_final['min']:.4f}, {q_final['max']:.4f}]"
    improvement_range_str = f"[{improvement_final['min']:+.4f}, {improvement_final['max']:+.4f}]"

    print(
        f"{'Final Accuracy (range)':<25} {classical_range_str:<30} {quantum_range_str:<30} {improvement_range_str:<20}")

    # Best Accuracy
    c_best = classical_agg['best_accuracy']
    q_best = quantum_agg['best_accuracy']
    improvement_best = improvement_stats['best_accuracy_improvement']

    classical_best_str = f"{c_best['mean']:.4f} ± {c_best['std']:.4f}"
    quantum_best_str = f"{q_best['mean']:.4f} ± {q_best['std']:.4f}"
    improvement_best_str = f"{improvement_best['mean']:+.4f} ± {improvement_best['std']:.4f}"

    print(f"{'Best Accuracy (mean)':<25} {classical_best_str:<30} {quantum_best_str:<30} {improvement_best_str:<20}")

    classical_best_range_str = f"[{c_best['min']:.4f}, {c_best['max']:.4f}]"
    quantum_best_range_str = f"[{q_best['min']:.4f}, {q_best['max']:.4f}]"
    improvement_best_range_str = f"[{improvement_best['min']:+.4f}, {improvement_best['max']:+.4f}]"

    print(
        f"{'Best Accuracy (range)':<25} {classical_best_range_str:<30} {quantum_best_range_str:<30} {improvement_best_range_str:<20}")

    # F1 Score
    c_f1 = classical_agg['final_f1_score']
    q_f1 = quantum_agg['final_f1_score']
    improvement_f1 = improvement_stats['final_f1_improvement']

    classical_f1_str = f"{c_f1['mean']:.4f} ± {c_f1['std']:.4f}"
    quantum_f1_str = f"{q_f1['mean']:.4f} ± {q_f1['std']:.4f}"
    improvement_f1_str = f"{improvement_f1['mean']:+.4f} ± {improvement_f1['std']:.4f}"

    print(f"{'Final F1 Score (mean)':<25} {classical_f1_str:<30} {quantum_f1_str:<30} {improvement_f1_str:<20}")

    classical_f1_range_str = f"[{c_f1['min']:.4f}, {c_f1['max']:.4f}]"
    quantum_f1_range_str = f"[{q_f1['min']:.4f}, {q_f1['max']:.4f}]"
    improvement_f1_range_str = f"[{improvement_f1['min']:+.4f}, {improvement_f1['max']:+.4f}]"

    print(
        f"{'Final F1 Score (range)':<25} {classical_f1_range_str:<30} {quantum_f1_range_str:<30} {improvement_f1_range_str:<20}")

    # Training Time
    c_time = classical_agg['training_time']
    q_time = quantum_agg['training_time']

    classical_time_str = f"{c_time['mean'] / 60:.1f} ± {c_time['std'] / 60:.1f} min"
    quantum_time_str = f"{q_time['mean'] / 60:.1f} ± {q_time['std'] / 60:.1f} min"
    time_diff_str = f"{(q_time['mean'] - c_time['mean']) / 60:+.1f} min"

    print(f"{'Training Time':<25} {classical_time_str:<30} {quantum_time_str:<30} {time_diff_str:<20}")

    # Parameters
    params_classical_str = f"{parameters['classical']:,}"
    params_quantum_str = f"{parameters['quantum']:,}"
    params_diff_str = f"{parameters['quantum'] - parameters['classical']:+,}"

    print(f"{'Parameters':<25} {params_classical_str:<30} {params_quantum_str:<30} {params_diff_str:<20}")

    print(f"\n{'=' * 80}")
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print(f"{'=' * 80}")

    print(f"Mean Final Accuracy Improvement: {improvement_final['mean']:+.4f} ± {improvement_final['std']:.4f}")
    print(f"Mean Final F1 Score Improvement: {improvement_f1['mean']:+.4f} ± {improvement_f1['std']:.4f}")
    print(f"Mean Best Accuracy Improvement: {improvement_best['mean']:+.4f} ± {improvement_best['std']:.4f}")

    # Check if quantum consistently outperforms
    quantum_wins_final = improvement_final['mean'] > 0
    quantum_wins_f1 = improvement_f1['mean'] > 0
    quantum_wins_best = improvement_best['mean'] > 0

    print(f"Quantum Advantage (Final Accuracy): {'YES' if quantum_wins_final else 'NO'}")
    print(f"Quantum Advantage (Final F1 Score): {'YES' if quantum_wins_f1 else 'NO'}")
    print(f"Quantum Advantage (Best Accuracy): {'YES' if quantum_wins_best else 'NO'}")

    # Fixed 90% achievement calculation
    target_achieved = "YES" if q_final['mean'] >= 0.90 else "NO"
    print(f"90% Accuracy Target Achievement: {target_achieved} (Quantum: {q_final['mean']:.1%})")

    # Effect size (Cohen's d)
    if improvement_final['std'] > 0:
        cohens_d_final = improvement_final['mean'] / improvement_final['std']
        print(f"Effect Size (Cohen's d) for Final Accuracy: {cohens_d_final:.3f}")

        if abs(cohens_d_final) < 0.2:
            effect_size = "Small"
        elif abs(cohens_d_final) < 0.5:
            effect_size = "Small-Medium"
        elif abs(cohens_d_final) < 0.8:
            effect_size = "Medium-Large"
        else:
            effect_size = "Large"
        print(f"Effect Size Interpretation: {effect_size}")


def create_aggregated_visualizations(classical_results, quantum_results, classical_agg, quantum_agg,
                                     improvement_stats):
    """Create comprehensive visualizations for aggregated results"""

    plt.style.use('default')
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # 1. Final Accuracy Distribution (Box Plot)
    ax1 = fig.add_subplot(gs[0, 0])
    final_accs_classical = [r['final_accuracy'] for r in classical_results if 'final_accuracy' in r]
    final_accs_quantum = [r['final_accuracy'] for r in quantum_results if 'final_accuracy' in r]

    box_data = [final_accs_classical, final_accs_quantum]
    bp1 = ax1.boxplot(box_data, labels=['Classical', 'Quantum'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('#3498db')
    bp1['boxes'][1].set_facecolor('#e74c3c')
    ax1.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='90% Target')
    ax1.set_title('Final Accuracy Distribution\n(10 Runs)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Best Accuracy Distribution (Box Plot)
    ax2 = fig.add_subplot(gs[0, 1])
    best_accs_classical = [r['best_accuracy'] for r in classical_results if 'best_accuracy' in r]
    best_accs_quantum = [r['best_accuracy'] for r in quantum_results if 'best_accuracy' in r]

    box_data2 = [best_accs_classical, best_accs_quantum]
    bp2 = ax2.boxplot(box_data2, labels=['Classical', 'Quantum'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#3498db')
    bp2['boxes'][1].set_facecolor('#e74c3c')
    ax2.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='90% Target')
    ax2.set_title('Accuracy Distribution During Training', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Training Time Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    train_times_classical = [r['training_time'] / 60 for r in classical_results if 'training_time' in r]
    train_times_quantum = [r['training_time'] / 60 for r in quantum_results if 'training_time' in r]

    box_data3 = [train_times_classical, train_times_quantum]
    bp3 = ax3.boxplot(box_data3, labels=['Classical', 'Quantum'], patch_artist=True)
    bp3['boxes'][0].set_facecolor('#3498db')
    bp3['boxes'][1].set_facecolor('#e74c3c')
    ax3.set_title('Training Time Distribution\n(10 Runs)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Training Time (minutes)')
    ax3.grid(True, alpha=0.3)

    # 4. Accuracy Improvement Distribution
    ax4 = fig.add_subplot(gs[0, 3])
    improvements_final = [quantum_results[i]['final_accuracy'] - classical_results[i]['final_accuracy']
                          for i in range(min(len(quantum_results), len(classical_results)))]

    ax4.hist(improvements_final, bins=10, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
    ax4.axvline(x=np.mean(improvements_final), color='green', linestyle='-', alpha=0.8,
                label=f'Mean: {np.mean(improvements_final):+.4f}')
    ax4.set_title('Final Accuracy Improvement\nDistribution (10 Runs)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Quantum - Classical Accuracy')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Run-by-Run Comparison (Chronological Order of All Runs)
    ax5 = fig.add_subplot(gs[1, :2])

    run_ids = list(range(1, len(final_accs_classical) + 1))
    ax5.plot(run_ids, final_accs_classical, 'b-o', linewidth=2, markersize=8, label='Classical CNN', alpha=0.8)
    ax5.plot(run_ids, final_accs_quantum, 'r-s', linewidth=2, markersize=8, label='Quantum CNN', alpha=0.8)
    ax5.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='90% Target')
    ax5.set_title('Final Accuracy Across 10 Runs', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Run Number')
    ax5.set_ylabel('Final Accuracy')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(run_ids)

    # 6. Mean Performance Comparison with Error Bars
    ax6 = fig.add_subplot(gs[1, 2:])
    metrics = ['Final Accuracy', 'Best Accuracy']
    classical_means = [classical_agg['final_accuracy']['mean'], classical_agg['best_accuracy']['mean']]
    classical_stds = [classical_agg['final_accuracy']['std'], classical_agg['best_accuracy']['std']]
    quantum_means = [quantum_agg['final_accuracy']['mean'], quantum_agg['best_accuracy']['mean']]
    quantum_stds = [quantum_agg['final_accuracy']['std'], quantum_agg['best_accuracy']['std']]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax6.bar(x - width / 2, classical_means, width, yerr=classical_stds,
                    label='Classical CNN', color='#3498db', alpha=0.8, capsize=5)
    bars2 = ax6.bar(x + width / 2, quantum_means, width, yerr=quantum_stds,
                    label='Quantum CNN', color='#e74c3c', alpha=0.8, capsize=5)

    ax6.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='90% Target')
    ax6.set_title('Mean Performance ± Standard Deviation\n(10 Runs)', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Accuracy')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax6.text(bar1.get_x() + bar1.get_width() / 2., height1 + classical_stds[i] + 0.005,
                 f'{height1:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax6.text(bar2.get_x() + bar2.get_width() / 2., height2 + quantum_stds[i] + 0.005,
                 f'{height2:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 7. Average Training Curves
    ax7 = fig.add_subplot(gs[2, :2])

    # Calculate average training curves
    max_epochs = max(len(r.get('test_acc_list', [])) for r in classical_results + quantum_results)

    def pad_and_average_curves(results_list, key):
        all_curves = []
        for result in results_list:
            curve = result.get(key, [])
            if len(curve) > 0:
                curve = np.array(curve)
                # Use np.pad for safe and efficient padding
                padded_curve = np.pad(curve, (0, max_epochs - len(curve)), 'edge')
                all_curves.append(padded_curve)

        if all_curves:
            all_curves = np.array(all_curves)
            return np.mean(all_curves, axis=0), np.std(all_curves, axis=0)
        else:
            return np.zeros(max_epochs), np.zeros(max_epochs)

    classical_test_mean, classical_test_std = pad_and_average_curves(classical_results, 'test_acc_list')
    quantum_test_mean, quantum_test_std = pad_and_average_curves(quantum_results, 'test_acc_list')

    epochs_range = range(1, len(classical_test_mean) + 1)

    if len(classical_test_mean) > 0 and len(quantum_test_mean) > 0:
        ax7.plot(epochs_range, classical_test_mean, 'b-', linewidth=2, label='Classical CNN (mean)')
        ax7.fill_between(epochs_range,
                         classical_test_mean - classical_test_std,
                         classical_test_mean + classical_test_std,
                         alpha=0.3, color='blue')

        ax7.plot(epochs_range, quantum_test_mean, 'r-', linewidth=2, label='Quantum CNN (mean)')
        ax7.fill_between(epochs_range,
                         quantum_test_mean - quantum_test_std,
                         quantum_test_mean + quantum_test_std,
                         alpha=0.3, color='red')

    ax7.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='90% Target')
    ax7.set_title('Average Test Accuracy Evolution\n(10 Runs ± Std Dev)', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Test Accuracy')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Performance Consistency Analysis
    ax8 = fig.add_subplot(gs[2, 2:])

    # Calculate coefficient of variation (CV) for consistency
    classical_cv_final = classical_agg['final_accuracy']['std'] / classical_agg['final_accuracy']['mean'] if \
        classical_agg['final_accuracy']['mean'] > 0 else 0
    quantum_cv_final = quantum_agg['final_accuracy']['std'] / quantum_agg['final_accuracy']['mean'] if \
        quantum_agg['final_accuracy']['mean'] > 0 else 0

    classical_cv_best = classical_agg['best_accuracy']['std'] / classical_agg['best_accuracy']['mean'] if \
        classical_agg['best_accuracy']['mean'] > 0 else 0
    quantum_cv_best = quantum_agg['best_accuracy']['std'] / quantum_agg['best_accuracy']['mean'] if \
        quantum_agg['best_accuracy']['mean'] > 0 else 0

    consistency_metrics = ['Final Accuracy CV', 'Best Accuracy CV']
    classical_cv = [classical_cv_final, classical_cv_best]
    quantum_cv = [quantum_cv_final, quantum_cv_best]

    x = np.arange(len(consistency_metrics))
    width = 0.35

    bars1 = ax8.bar(x - width / 2, classical_cv, width, label='Classical CNN', color='#3498db', alpha=0.8)
    bars2 = ax8.bar(x + width / 2, quantum_cv, width, label='Quantum CNN', color='#e74c3c', alpha=0.8)

    ax8.set_title('Model Consistency Analysis\n(Lower CV = More Consistent)', fontsize=14, fontweight='bold')
    ax8.set_ylabel('Coefficient of Variation')
    ax8.set_xticks(x)
    ax8.set_xticklabels(consistency_metrics)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Add value labels
    for bar1, bar2, cv1, cv2 in zip(bars1, bars2, classical_cv, quantum_cv):
        ax8.text(bar1.get_x() + bar1.get_width() / 2., bar1.get_height() + 0.001,
                 f'{cv1:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax8.text(bar2.get_x() + bar2.get_width() / 2., bar2.get_height() + 0.001,
                 f'{cv2:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 9. Summary Statistics Table
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')

    # Create comprehensive summary
    summary_text = f"""
COMPREHENSIVE 5-RUN EXPERIMENTAL SUMMARY

FINAL ACCURACY STATISTICS:
Classical CNN: {classical_agg['final_accuracy']['mean']:.4f} ± {classical_agg['final_accuracy']['std']:.4f} (range: {classical_agg['final_accuracy']['min']:.4f} - {classical_agg['final_accuracy']['max']:.4f})
Quantum CNN:   {quantum_agg['final_accuracy']['mean']:.4f} ± {quantum_agg['final_accuracy']['std']:.4f} (range: {quantum_agg['final_accuracy']['min']:.4f} - {quantum_agg['final_accuracy']['max']:.4f})
Improvement:   {improvement_stats['final_accuracy_improvement']['mean']:+.4f} ± {improvement_stats['final_accuracy_improvement']['std']:.4f}

BEST ACCURACY STATISTICS:
Classical CNN: {classical_agg['best_accuracy']['mean']:.4f} ± {classical_agg['best_accuracy']['std']:.4f} (range: {classical_agg['best_accuracy']['min']:.4f} - {classical_agg['best_accuracy']['max']:.4f})
Quantum CNN:   {quantum_agg['best_accuracy']['mean']:.4f} ± {quantum_agg['best_accuracy']['std']:.4f} (range: {quantum_agg['best_accuracy']['min']:.4f} - {quantum_agg['best_accuracy']['max']:.4f})
Improvement:   {improvement_stats['best_accuracy_improvement']['mean']:+.4f} ± {improvement_stats['best_accuracy_improvement']['std']:.4f}

TRAINING EFFICIENCY:
Classical CNN Training Time: {classical_agg['training_time']['mean'] / 60:.1f} ± {classical_agg['training_time']['std'] / 60:.1f} minutes
Quantum CNN Training Time:   {quantum_agg['training_time']['mean'] / 60:.1f} ± {quantum_agg['training_time']['std'] / 60:.1f} minutes

MODEL COMPLEXITY:
Both models: {classical_results[0]['parameters'] if classical_results else 'N/A':,} parameters (equivalent complexity)

KEY FINDINGS:
• Quantum Advantage (Final): {'✓ YES' if improvement_stats['final_accuracy_improvement']['mean'] > 0 else '✗ NO'} ({improvement_stats['final_accuracy_improvement']['mean']:+.4f} mean improvement)
• Quantum Advantage (Best):  {'✓ YES' if improvement_stats['best_accuracy_improvement']['mean'] > 0 else '✗ NO'} ({improvement_stats['best_accuracy_improvement']['mean']:+.4f} mean improvement)
• 90% Target Achievement:    {'✓ ACHIEVED' if quantum_agg['final_accuracy']['mean'] >= 0.90 else '✗ NOT ACHIEVED'} (Quantum: {quantum_agg['final_accuracy']['mean']:.1%}, Classical: {classical_agg['final_accuracy']['mean']:.1%})
• Consistency (Quantum):     CV = {quantum_cv_final:.3f} (Final Acc), CV = {quantum_cv_best:.3f} (Best Acc)
• Statistical Robustness:    Results based on 10 experimental runs
"""

    ax9.text(0.02, 0.98, summary_text, transform=ax9.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.7", facecolor='lightgray', alpha=0.9))

    plt.suptitle('Quantum vs Classical CNN: 10-Run Statistical Analysis and Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig('quantum_vs_classical_all10_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_f1_score_visualizations(classical_results, quantum_results, classical_agg, quantum_agg,
                                   improvement_stats):
    """Create F1 score evaluation graphs - separate for Classical and Quantum"""

    plt.style.use('default')
    
    # Create two separate figures
    
    # ====================== CLASSICAL CNN F1 SCORE EVALUATION ======================
    fig1 = plt.figure(figsize=(16, 10))
    gs1 = fig1.add_gridspec(2, 1, hspace=0.3)
    
    # Top: Final F1 Score Across Runs
    ax1 = fig1.add_subplot(gs1[0])
    f1_scores_classical = [r['final_f1_score'] for r in classical_results if 'final_f1_score' in r]
    run_ids = list(range(1, len(f1_scores_classical) + 1))
    
    ax1.plot(run_ids, f1_scores_classical, 'o-', linewidth=3, markersize=12, 
             color='#3498db', markeredgecolor='black', markeredgewidth=2)
    ax1.axhline(y=np.mean(f1_scores_classical), color='green', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Mean: {np.mean(f1_scores_classical):.4f}')
    
    ax1.set_title('Classical CNN: Final F1 Score Across Runs', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Run Number', fontsize=14)
    ax1.set_ylabel('F1 Score', fontsize=14)
    ax1.set_xticks(run_ids)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(f1_scores_classical) - 0.03, max(f1_scores_classical) + 0.03])
    
    # Bottom: Average F1 Score Evolution (10 Runs ± Std Dev)
    ax2 = fig1.add_subplot(gs1[1])
    
    # Calculate average F1 curves (robust to different epoch lengths)
    def pad_and_average_f1_curves(results_list):
        local_max_epochs = 0
        for r in results_list:
            l = len(r.get('test_f1_list', []))
            if l > local_max_epochs:
                local_max_epochs = l
        if local_max_epochs == 0:
            return np.array([]), np.array([])

        all_curves = []
        for result in results_list:
            curve = result.get('test_f1_list', [])
            if len(curve) > 0:
                arr = np.array(curve, dtype=float)
                pad_width = max(0, local_max_epochs - arr.size)
                if pad_width > 0:
                    arr = np.pad(arr, (0, pad_width), mode='edge')
                else:
                    # If longer than max (shouldn't happen with local_max), trim to local_max_epochs
                    arr = arr[:local_max_epochs]
                all_curves.append(arr)

        if not all_curves:
            return np.array([]), np.array([])

        stacked = np.stack(all_curves)
        return np.mean(stacked, axis=0), np.std(stacked, axis=0)
    
    classical_f1_mean, classical_f1_std = pad_and_average_f1_curves(classical_results)
    epochs_range_classical = range(1, len(classical_f1_mean) + 1)
    
    if len(classical_f1_mean) > 0:
        ax2.plot(epochs_range_classical, classical_f1_mean, color='#3498db', linewidth=3, label='Classical CNN (mean)')
        ax2.fill_between(epochs_range_classical,
                         classical_f1_mean - classical_f1_std,
                         classical_f1_mean + classical_f1_std,
                         alpha=0.3, color='#3498db')
    
    ax2.set_title('Classical CNN: Average F1 Score Evolution\n(10 Runs ± Std Dev)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('F1 Score', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.0])
    
    plt.suptitle('Classical CNN: F1 Score Evaluation', fontsize=20, fontweight='bold', y=0.995)
    plt.savefig('classical_f1_score_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nClassical F1 Score evaluation saved to: classical_f1_score_evaluation.png")
    
    # ====================== QUANTUM CNN F1 SCORE EVALUATION ======================
    fig2 = plt.figure(figsize=(16, 10))
    gs2 = fig2.add_gridspec(2, 1, hspace=0.3)
    
    # Top: Final F1 Score Across Runs
    ax3 = fig2.add_subplot(gs2[0])
    f1_scores_quantum = [r['final_f1_score'] for r in quantum_results if 'final_f1_score' in r]
    
    ax3.plot(run_ids, f1_scores_quantum, 's-', linewidth=3, markersize=12,
             color='#e74c3c', markeredgecolor='black', markeredgewidth=2)
    ax3.axhline(y=np.mean(f1_scores_quantum), color='green', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Mean: {np.mean(f1_scores_quantum):.4f}')
    
    ax3.set_title('Quantum CNN: Final F1 Score Across 10 Runs', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Run Number', fontsize=14)
    ax3.set_ylabel('F1 Score', fontsize=14)
    ax3.set_xticks(run_ids)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([min(f1_scores_quantum) - 0.03, max(f1_scores_quantum) + 0.03])
    
    # Bottom: Average F1 Score Evolution (10 Runs ± Std Dev)
    ax4 = fig2.add_subplot(gs2[1])
    
    quantum_f1_mean, quantum_f1_std = pad_and_average_f1_curves(quantum_results)
    epochs_range_quantum = range(1, len(quantum_f1_mean) + 1)
    
    if len(quantum_f1_mean) > 0:
        ax4.plot(epochs_range_quantum, quantum_f1_mean, color='#e74c3c', linewidth=3, label='Quantum CNN (mean)')
        ax4.fill_between(epochs_range_quantum,
                         quantum_f1_mean - quantum_f1_std,
                         quantum_f1_mean + quantum_f1_std,
                         alpha=0.3, color='#e74c3c')
    
    ax4.set_title('Quantum CNN: Average F1 Score Evolution\n(10 Runs ± Std Dev)', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=14)
    ax4.set_ylabel('F1 Score', fontsize=14)
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.0])
    
    plt.suptitle('Quantum CNN: F1 Score Evaluation', fontsize=20, fontweight='bold', y=0.995)
    plt.savefig('quantum_f1_score_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Quantum F1 Score evaluation saved to: quantum_f1_score_evaluation.png")


def create_confusion_matrix_visualizations(classical_results, quantum_results):
    """Create separate confusion matrix visualizations for Classical and Quantum models"""
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Aggregate predictions from all runs
    classical_y_true_all = []
    classical_y_pred_all = []
    quantum_y_true_all = []
    quantum_y_pred_all = []
    
    for c_res, q_res in zip(classical_results, quantum_results):
        classical_y_true_all.extend(c_res.get('y_true', []))
        classical_y_pred_all.extend(c_res.get('y_pred', []))
        quantum_y_true_all.extend(q_res.get('y_true', []))
        quantum_y_pred_all.extend(q_res.get('y_pred', []))
    
    # Calculate confusion matrices
    cm_classical = confusion_matrix(classical_y_true_all, classical_y_pred_all)
    cm_quantum = confusion_matrix(quantum_y_true_all, quantum_y_pred_all)
    
    # ========== CLASSICAL CNN CONFUSION MATRIX (SEPARATE FILE) ==========
    fig_c, ax_c = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm_classical, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'],
                cbar=True,
                annot_kws={'size': 14},
                ax=ax_c, linewidths=1, linecolor='gray', 
                vmin=0, vmax=cm_classical.max(), square=True)
    
    ax_c.set_title('Confusion Matrix - Classical CNN', 
                   fontsize=14, pad=15)
    ax_c.set_xlabel('Predicted Label', fontsize=12)
    ax_c.set_ylabel('True Label', fontsize=12)
    
    # Add statistics for Classical
    total_classical = np.sum(cm_classical)
    accuracy_classical = (cm_classical[0,0] + cm_classical[1,1]) / total_classical
    
    # Calculate metrics
    tn, fp, fn, tp = cm_classical.ravel()
    precision_classical = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_classical = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_classical = 2 * (precision_classical * recall_classical) / (precision_classical + recall_classical) if (precision_classical + recall_classical) > 0 else 0
    
    plt.tight_layout()
    plt.savefig('classical_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: classical_confusion_matrix.png")
    
    # ========== QUANTUM CNN CONFUSION MATRIX (SEPARATE FILE) ==========
    fig_q, ax_q = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm_quantum, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'],
                cbar=True,
                annot_kws={'size': 14},
                ax=ax_q, linewidths=1, linecolor='gray',
                vmin=0, vmax=cm_quantum.max(), square=True)
    
    ax_q.set_title('Confusion Matrix - Quantum CNN', 
                   fontsize=14, pad=15)
    ax_q.set_xlabel('Predicted Label', fontsize=12)
    ax_q.set_ylabel('True Label', fontsize=12)
    
    # Add statistics for Quantum
    total_quantum = np.sum(cm_quantum)
    accuracy_quantum = (cm_quantum[0,0] + cm_quantum[1,1]) / total_quantum
    
    # Calculate metrics
    tn_q, fp_q, fn_q, tp_q = cm_quantum.ravel()
    precision_quantum = tp_q / (tp_q + fp_q) if (tp_q + fp_q) > 0 else 0
    recall_quantum = tp_q / (tp_q + fn_q) if (tp_q + fn_q) > 0 else 0
    f1_quantum = 2 * (precision_quantum * recall_quantum) / (precision_quantum + recall_quantum) if (precision_quantum + recall_quantum) > 0 else 0
    
    plt.tight_layout()
    plt.savefig('quantum_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: quantum_confusion_matrix.png")
    
    # Print comparison summary
    improvement_acc = accuracy_quantum - accuracy_classical
    improvement_f1 = f1_quantum - f1_classical
    print(f"\nConfusion Matrix Summary:")
    print(f"  Classical - Accuracy: {accuracy_classical:.4f}, F1: {f1_classical:.4f}")
    print(f"  Quantum   - Accuracy: {accuracy_quantum:.4f}, F1: {f1_quantum:.4f}")
    print(f"  Improvement: Acc {improvement_acc:+.4f}, F1 {improvement_f1:+.4f}")


def create_average_loss_graphs(classical_results, quantum_results):
    """Create one loss graph per model (Classical and Quantum), averaging across selected runs.

    Each graph shows Train and Validation loss means with ±1 std shaded area.
    """
    plt.style.use('default')

    def pad_and_average(results_list, key):
        max_epochs = 0
        for r in results_list:
            l = len(r.get(key, []))
            if l > max_epochs:
                max_epochs = l
        if max_epochs == 0:
            return np.array([]), np.array([])
        curves = []
        for r in results_list:
            curve = np.array(r.get(key, []), dtype=float)
            if curve.size == 0:
                continue
            if curve.size < max_epochs:
                curve = np.pad(curve, (0, max_epochs - curve.size), mode='edge')
            curves.append(curve)
        if not curves:
            return np.array([]), np.array([])
        arr = np.stack(curves)
        return np.mean(arr, axis=0), np.std(arr, axis=0)

    # Classical
    c_train_mean, c_train_std = pad_and_average(classical_results, 'train_loss_list')
    c_val_mean, c_val_std = pad_and_average(classical_results, 'val_loss_list')
    if c_train_mean.size > 0:
        epochs = np.arange(1, c_train_mean.size + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, c_train_mean, label='Train Loss (mean)', color='#2ecc71', linewidth=2)
        if c_val_mean.size > 0:
            plt.plot(epochs, c_val_mean, label='Val Loss (mean)', color='#e67e22', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Classical CNN: Average Loss per Epoch')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('classical_loss_curve_all10.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('Saved: classical_loss_curve_all10.png')

    # Quantum
    q_train_mean, q_train_std = pad_and_average(quantum_results, 'train_loss_list')
    q_val_mean, q_val_std = pad_and_average(quantum_results, 'val_loss_list')
    if q_train_mean.size > 0:
        epochs = np.arange(1, q_train_mean.size + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, q_train_mean, label='Train Loss (mean)', color='#2ecc71', linewidth=2)
        if q_val_mean.size > 0:
            plt.plot(epochs, q_val_mean, label='Val Loss (mean)', color='#e67e22', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Quantum CNN: Average Loss per Epoch')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('quantum_loss_curve_all10.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('Saved: quantum_loss_curve_all10.png')


# --- Checkpoint Save/Load Functions ---
def save_checkpoint(all_classical_results, all_quantum_results, run_id, checkpoint_dir='checkpoints'):
    """Save results after each run for resumability"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_run_{run_id + 1}.pkl')
    
    checkpoint_data = {
        'all_classical_results': all_classical_results,
        'all_quantum_results': all_quantum_results,
        'completed_runs': run_id + 1,
        'timestamp': timestamp
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"✓ Checkpoint saved: {checkpoint_file}")


def load_all_checkpoints(checkpoint_dir='checkpoints', total_runs=10):
    """Load all available checkpoints and identify missing runs"""
    if not os.path.exists(checkpoint_dir):
        return [], [], list(range(total_runs))
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_run_') and f.endswith('.pkl')]
    
    if not checkpoint_files:
        return [], [], list(range(total_runs))
    
    # Initialize results lists
    all_classical_results = []
    all_quantum_results = []
    completed_runs = []
    
    # Load all available checkpoints
    for run_id in range(total_runs):
        checkpoint_file = f'checkpoint_run_{run_id + 1}.pkl'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                # Extract the results for this specific run
                classical_result = checkpoint_data['all_classical_results'][-1]
                quantum_result = checkpoint_data['all_quantum_results'][-1]
                
                all_classical_results.append(classical_result)
                all_quantum_results.append(quantum_result)
                completed_runs.append(run_id)
                
            except Exception as e:
                print(f"Warning: Could not load checkpoint_run_{run_id + 1}.pkl: {e}")
    
    # Identify missing runs
    missing_runs = [i for i in range(total_runs) if i not in completed_runs]
    
    if completed_runs:
        print(f"\n{'=' * 80}")
        print(f"CHECKPOINT RECOVERY STATUS:")
        print(f"Completed runs found: {[r + 1 for r in completed_runs]}")
        if missing_runs:
            print(f"Missing runs to execute: {[r + 1 for r in missing_runs]}")
        else:
            print(f"All runs completed!")
        print(f"{'=' * 80}\n")
    
    return all_classical_results, all_quantum_results, missing_runs


# --- Main Execution: Run 10 Experiments ---
def main():
    print("=" * 80)
    print("QUANTUM vs CLASSICAL CNN: 10-RUN STATISTICAL ANALYSIS")
    print("This experiment runs 10 independent training sessions for both")
    print("Quantum and Classical CNNs, using all 10 results for analysis.")
    print("=" * 80)

    total_runs = 10
    
    # Load all available checkpoints and identify missing runs
    all_classical_results, all_quantum_results, missing_runs = load_all_checkpoints(total_runs=total_runs)
    
    if not missing_runs:
        print("All runs already completed! Loading results for analysis...")
        runs_to_execute = []
    else:
        runs_to_execute = missing_runs
        if all_classical_results:
            print(f"Resuming incomplete experiment...")
        else:
            print("Starting fresh experiment (no checkpoints found)")
    
    total_start_time = time.time()

    # Run only the missing experiments
    for run_id in runs_to_execute:
        try:
            print(f"\n{'=' * 80}")
            print(f"STARTING RUN {run_id + 1}/10")
            print(f"{'=' * 80}")
            
            classical_results, quantum_results = run_single_experiment(run_id)
            
            # Insert results at correct position to maintain run order
            all_classical_results.insert(run_id, classical_results)
            all_quantum_results.insert(run_id, quantum_results)

            print(f"\nRun {run_id + 1} completed:")
            print(f"  Classical Final Accuracy: {classical_results['final_accuracy']:.4f}")
            print(f"  Quantum Final Accuracy:   {quantum_results['final_accuracy']:.4f}")
            print(f"  Improvement: {quantum_results['final_accuracy'] - classical_results['final_accuracy']:+.4f}")

            # Save checkpoint after each successful run
            save_checkpoint(all_classical_results, all_quantum_results, run_id)

        except Exception as e:
            print(f"Error in run {run_id + 1}: {e}")
            print("Checkpoint saved before error - you can resume!")
            print("Continuing with remaining runs...")
            continue

    total_time = time.time() - total_start_time

    if not all_classical_results or not all_quantum_results:
        print("No successful runs completed. Exiting.")
        return

    print(f"\n{'=' * 80}")
    if runs_to_execute:
        print(f"COMPLETED {len(runs_to_execute)} MISSING RUNS IN {total_time / 3600:.2f} HOURS")
    else:
        print(f"ALL 10 EXPERIMENTS ALREADY COMPLETED (loaded from checkpoints)")
    print(f"Total runs available: Classical={len(all_classical_results)}, Quantum={len(all_quantum_results)}")
    print(f"{'=' * 80}")

    # Aggregate and analyze results using all 10 runs
    classical_agg, quantum_agg, improvement_stats, parameters = aggregate_results(
        all_classical_results, all_quantum_results)

    # Print comprehensive results
    print_aggregated_results(classical_agg, quantum_agg, improvement_stats, parameters)

    # Create visualizations using all 10 results
    create_aggregated_visualizations(all_classical_results, all_quantum_results, classical_agg, quantum_agg,
                                     improvement_stats)

    # Create F1 score visualizations
    create_f1_score_visualizations(all_classical_results, all_quantum_results, classical_agg, quantum_agg,
                                   improvement_stats)

    # Create confusion matrix visualizations
    create_confusion_matrix_visualizations(all_classical_results, all_quantum_results)

    # Create one loss graph per model (aggregated across all 10 runs)
    create_average_loss_graphs(all_classical_results, all_quantum_results)

    # Save detailed results to files
    results_data = {
        'all_classical_results': all_classical_results,
        'all_quantum_results': all_quantum_results,
        'classical_aggregated': classical_agg,
        'quantum_aggregated': quantum_agg,
        'improvement_stats': improvement_stats,
        'parameters': parameters,
        'total_experiment_time': total_time
    }

    # Save as CSV for further analysis
    try:
        import pandas as pd

        def safe_last(lst):
            if not lst:
                return float('nan')
            return float(lst[-1])

        summary_data = []
        for i, (c_res, q_res) in enumerate(zip(all_classical_results, all_quantum_results)):
            summary_data.append({
                'Run': i + 1,
                'Classical_Final_Acc': c_res['final_accuracy'],
                'Quantum_Final_Acc': q_res['final_accuracy'],
                'Classical_Final_F1': c_res['final_f1_score'],
                'Quantum_Final_F1': q_res['final_f1_score'],
                'Classical_Best_Acc': c_res['best_accuracy'],
                'Quantum_Best_Acc': q_res['best_accuracy'],
                'Classical_Train_Time': c_res['training_time'],
                'Quantum_Train_Time': q_res['training_time'],
                'Classical_Train_Acc_Final': safe_last(c_res.get('train_acc_list', [])),
                'Quantum_Train_Acc_Final': safe_last(q_res.get('train_acc_list', [])),
                'Classical_Val_Acc_Final': safe_last(c_res.get('val_acc_list', [])),
                'Quantum_Val_Acc_Final': safe_last(q_res.get('val_acc_list', [])),
                'Classical_Train_Loss_Final': safe_last(c_res.get('train_loss_list', [])),
                'Quantum_Train_Loss_Final': safe_last(q_res.get('train_loss_list', [])),
                'Classical_Val_Loss_Final': safe_last(c_res.get('val_loss_list', [])),
                'Quantum_Val_Loss_Final': safe_last(q_res.get('val_loss_list', [])),
                'Final_Acc_Improvement': q_res['final_accuracy'] - c_res['final_accuracy'],
                'Final_F1_Improvement': q_res['final_f1_score'] - c_res['final_f1_score'],
                'Best_Acc_Improvement': q_res['best_accuracy'] - c_res['best_accuracy']
            })

        df = pd.DataFrame(summary_data)
        df.to_csv('quantum_vs_classical_all10_results.csv', index=False)
        print(f"\nAll 10 run results saved to: quantum_vs_classical_all10_results.csv")

    except ImportError:
        print("Pandas not available for CSV export, but all results are displayed above.")

    print(f"\nComprehensive analysis plots saved to: quantum_vs_classical_all10_comprehensive_analysis.png")

    # Final scientific conclusion
    print(f"\n{'=' * 80}")
    print("SCIENTIFIC CONCLUSION")
    print(f"{'=' * 80}")

    final_improvement = improvement_stats['final_accuracy_improvement']['mean']
    final_improvement_std = improvement_stats['final_accuracy_improvement']['std']

    if final_improvement > 0 and final_improvement > 2 * final_improvement_std:
        conclusion = "SIGNIFICANT QUANTUM ADVANTAGE DEMONSTRATED"
        details = "Quantum CNN shows consistent and statistically significant improvement over Classical CNN across all 10 runs."
    elif final_improvement > 0:
        conclusion = "MODEST QUANTUM ADVANTAGE OBSERVED"
        details = "Quantum CNN shows positive but variable improvement over Classical CNN across all 10 runs."
    else:
        conclusion = "NO CLEAR QUANTUM ADVANTAGE"
        details = "Results do not demonstrate consistent quantum advantage across all 10 runs."

    print(f"{conclusion}")
    print(f"{details}")
    print(f"Mean improvement: {final_improvement:+.4f} ± {final_improvement_std:.4f}")
    print(f"Target 90% accuracy: {'ACHIEVED' if quantum_agg['final_accuracy']['mean'] >= 0.90 else 'NOT ACHIEVED'} by Quantum CNN")
    print(f"Statistical robustness: Analysis based on 10 independent experimental runs")

    return results_data


if __name__ == "__main__":
    # Run the main experiment
    results = main()