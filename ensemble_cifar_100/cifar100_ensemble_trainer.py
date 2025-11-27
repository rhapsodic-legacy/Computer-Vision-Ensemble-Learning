"""
CIFAR-100 Ensemble Trainer
Train multiple diverse models for ensemble predictions

Strategy:
1. Train models with different architectures
2. Train with different random seeds
3. Train with different hyperparameters
4. Combine predictions for superior accuracy

Expected: 2-5% improvement over single model
Target: 74-77% accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Tuple, Dict, List
import random
import json
from datetime import datetime

# ============================================================================
# ENSEMBLE CONFIGURATIONS
# ============================================================================

class EnsembleConfig:
    """Configuration for ensemble training"""
    
    # Model diversity
    ARCHITECTURES = [
        'efficientnet_b0',
        'resnet50',
        'convnext_tiny',
    ]
    
    # Training diversity
    RANDOM_SEEDS = [42, 123, 456, 789, 2024]
    
    # Hyperparameter variations
    MIXUP_ALPHAS = [0.1, 0.2, 0.3]
    DROPOUTS = [0.2, 0.3, 0.4]
    INITIAL_LRS = [1e-3, 5e-4, 2e-3]
    
    # Training settings
    NUM_EPOCHS = 200  # Reduced for faster ensemble training
    BATCH_SIZE = 128
    NUM_CLASSES = 100
    WEIGHT_DECAY = 5e-4
    WARMUP_EPOCHS = 10
    MIN_LR = 1e-6
    PATIENCE = 25
    
    # Ensemble strategy
    NUM_MODELS = 5  # Number of models to train
    SELECTION_STRATEGY = 'diverse'  # 'diverse' or 'best'

config = EnsembleConfig()

# ============================================================================
# IMPORT UTILITIES FROM ENHANCED SCRIPT
# ============================================================================

def get_cifar100_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Get CIFAR-100 data loaders with strong augmentation"""
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), 
                           (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), 
                           (0.2675, 0.2565, 0.2761)),
    ])
    
    train_dataset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    return train_loader, test_loader

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Apply MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """Apply CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp/CutMix loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(pred, dim=-1)
        
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class CIFAR100Classifier(nn.Module):
    """CIFAR-100 classifier with attention"""
    
    def __init__(self, num_classes: int = 100, model_name: str = 'efficientnet_b0', 
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
        if model_name == 'efficientnet_b0':
            weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = torchvision.models.efficientnet_b0(weights=weights)
            feat_dim = 1280
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        elif model_name == 'resnet50':
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = torchvision.models.resnet50(weights=weights)
            feat_dim = 2048
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        elif model_name == 'convnext_tiny':
            weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = torchvision.models.convnext_tiny(weights=weights)
            feat_dim = 768
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.attention = SEBlock(feat_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.flatten(1)
        
        b, c = x.size()
        x_2d = x.view(b, c, 1, 1)
        x_2d = self.attention(x_2d)
        x = x_2d.flatten(1)
        
        x = self.classifier(x)
        return x

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup"""
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, 
                 initial_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
    
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def compute_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """Compute top-k accuracy"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# ============================================================================
# ENSEMBLE TRAINING
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha) -> Dict:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Apply MixUp or CutMix
        use_mixup = random.random() < 0.5
        if use_mixup and random.random() < 0.5:
            data, target_a, target_b, lam = mixup_data(data, target, mixup_alpha)
        elif use_mixup:
            data, target_a, target_b, lam = cutmix_data(data, target, 1.0)
        else:
            target_a, target_b, lam = target, target, 1.0
        
        optimizer.zero_grad()
        output = model(data)
        
        if use_mixup:
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if not use_mixup or lam == 1.0:
            acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
            correct_top1 += acc1.item() * target.size(0) / 100
            correct_top5 += acc5.item() * target.size(0) / 100
        total += target.size(0)
        
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct_top1 / total if total > 0 else 0
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'top1_acc': 100. * correct_top1 / total if total > 0 else 0,
        'top5_acc': 100. * correct_top5 / total if total > 0 else 0
    }

def validate(model, test_loader, criterion, device) -> Dict:
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Validation', leave=False)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
            correct_top1 += acc1.item() * target.size(0) / 100
            correct_top5 += acc5.item() * target.size(0) / 100
            total += target.size(0)
            
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100. * correct_top1 / total
            })
    
    return {
        'loss': total_loss / len(test_loader),
        'top1_acc': 100. * correct_top1 / total,
        'top5_acc': 100. * correct_top5 / total
    }

def train_single_model(model_id: int, architecture: str, random_seed: int,
                      mixup_alpha: float, dropout: float, initial_lr: float,
                      device, train_loader, test_loader) -> Dict:
    """Train a single model for the ensemble"""
    
    print(f"\n{'='*80}")
    print(f"Training Model #{model_id}")
    print(f"Architecture: {architecture} | Seed: {random_seed}")
    print(f"MixUp: {mixup_alpha} | Dropout: {dropout} | LR: {initial_lr}")
    print(f"{'='*80}\n")
    
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Create model
    model = CIFAR100Classifier(
        num_classes=config.NUM_CLASSES,
        model_name=architecture,
        pretrained=True,
        dropout=dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=config.NUM_EPOCHS,
        initial_lr=initial_lr,
        min_lr=config.MIN_LR
    )
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_top1_acc': []}
    
    for epoch in range(config.NUM_EPOCHS):
        current_lr = scheduler.step(epoch)
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha)
        val_metrics = validate(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_top1_acc'].append(val_metrics['top1_acc'])
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | "
                  f"Val Acc: {val_metrics['top1_acc']:.2f}% | "
                  f"Top-5: {val_metrics['top5_acc']:.2f}% | "
                  f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['top1_acc'] > best_val_acc:
            best_val_acc = val_metrics['top1_acc']
            patience_counter = 0
            
            # Save checkpoint
            model_path = f'ensemble_models/model_{model_id}_best.pth'
            torch.save({
                'model_id': model_id,
                'epoch': epoch,
                'architecture': architecture,
                'random_seed': random_seed,
                'mixup_alpha': mixup_alpha,
                'dropout': dropout,
                'initial_lr': initial_lr,
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'val_top5_acc': val_metrics['top5_acc'],
                'history': history
            }, model_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    print(f"\n✓ Model #{model_id} complete | Best Val Acc: {best_val_acc:.2f}%\n")
    
    return {
        'model_id': model_id,
        'architecture': architecture,
        'random_seed': random_seed,
        'mixup_alpha': mixup_alpha,
        'dropout': dropout,
        'initial_lr': initial_lr,
        'best_val_acc': best_val_acc,
        'model_path': model_path
    }

def select_model_configurations(num_models: int, strategy: str = 'diverse') -> List[Dict]:
    """Select diverse model configurations for ensemble"""
    
    configurations = []
    
    if strategy == 'diverse':
        # Create diverse configurations
        for i in range(num_models):
            config_dict = {
                'model_id': i + 1,
                'architecture': config.ARCHITECTURES[i % len(config.ARCHITECTURES)],
                'random_seed': config.RANDOM_SEEDS[i % len(config.RANDOM_SEEDS)],
                'mixup_alpha': config.MIXUP_ALPHAS[i % len(config.MIXUP_ALPHAS)],
                'dropout': config.DROPOUTS[i % len(config.DROPOUTS)],
                'initial_lr': config.INITIAL_LRS[i % len(config.INITIAL_LRS)]
            }
            configurations.append(config_dict)
    
    elif strategy == 'best':
        # Use best known configuration with different seeds
        for i in range(num_models):
            config_dict = {
                'model_id': i + 1,
                'architecture': config.ARCHITECTURES[0],  # Best architecture
                'random_seed': config.RANDOM_SEEDS[i % len(config.RANDOM_SEEDS)],
                'mixup_alpha': 0.2,  # Best mixup
                'dropout': 0.3,  # Best dropout
                'initial_lr': 1e-3  # Best LR
            }
            configurations.append(config_dict)
    
    return configurations

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_ensemble():
    """Train ensemble of models"""
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create directories
    os.makedirs('ensemble_models', exist_ok=True)
    
    # Load data
    print("\nLoading CIFAR-100 dataset...")
    train_loader, test_loader = get_cifar100_loaders(config.BATCH_SIZE)
    
    # Select model configurations
    print(f"\nGenerating {config.NUM_MODELS} diverse model configurations...")
    configurations = select_model_configurations(
        config.NUM_MODELS, 
        config.SELECTION_STRATEGY
    )
    
    # Print configurations
    print("\nModel Configurations:")
    print("="*80)
    for cfg in configurations:
        print(f"Model {cfg['model_id']}: {cfg['architecture']} | "
              f"Seed={cfg['random_seed']} | MixUp={cfg['mixup_alpha']} | "
              f"Drop={cfg['dropout']} | LR={cfg['initial_lr']}")
    print("="*80)
    
    # Train models
    results = []
    start_time = datetime.now()
    
    for cfg in configurations:
        result = train_single_model(
            model_id=cfg['model_id'],
            architecture=cfg['architecture'],
            random_seed=cfg['random_seed'],
            mixup_alpha=cfg['mixup_alpha'],
            dropout=cfg['dropout'],
            initial_lr=cfg['initial_lr'],
            device=device,
            train_loader=train_loader,
            test_loader=test_loader
        )
        results.append(result)
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 3600
    
    # Save ensemble metadata
    ensemble_info = {
        'num_models': config.NUM_MODELS,
        'strategy': config.SELECTION_STRATEGY,
        'training_time_hours': training_time,
        'models': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('ensemble_models/ensemble_info.json', 'w') as f:
        json.dump(ensemble_info, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING COMPLETE")
    print("="*80)
    print(f"\nTotal training time: {training_time:.2f} hours")
    print(f"Models trained: {config.NUM_MODELS}")
    print("\nIndividual Model Performance:")
    print("-"*80)
    
    accuracies = []
    for result in results:
        print(f"Model {result['model_id']} ({result['architecture']}): "
              f"{result['best_val_acc']:.2f}%")
        accuracies.append(result['best_val_acc'])
    
    print("-"*80)
    print(f"Best single model: {max(accuracies):.2f}%")
    print(f"Average accuracy: {np.mean(accuracies):.2f}%")
    print(f"Std deviation: {np.std(accuracies):.2f}%")
    print("\n✓ Models saved in ensemble_models/")
    print("✓ Run cifar100_ensemble_predictor.py to get ensemble predictions!")
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("CIFAR-100 ENSEMBLE TRAINER")
    print("="*80)
    print(f"\nTraining {config.NUM_MODELS} diverse models")
    print(f"Strategy: {config.SELECTION_STRATEGY}")
    print(f"Expected time: {config.NUM_MODELS * 3.5:.1f} hours (GPU)")
    print("\nPress Ctrl+C to cancel\n")
    
    results = train_ensemble()
