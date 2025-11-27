"""
CIFAR-100 Ensemble Predictor
Combine predictions from multiple trained models

Ensemble Methods:
1. Simple Averaging - Average all model predictions
2. Weighted Averaging - Weight by validation accuracy
3. Majority Voting - Vote on predictions
4. Rank-based Fusion - Combine based on confidence ranks

Expected: 2-5% boost over best single model
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
import json
from typing import List, Dict, Tuple
from collections import Counter

# ============================================================================
# MODEL ARCHITECTURE (copied from trainer)
# ============================================================================

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
    """CIFAR-100 classifier"""
    
    def __init__(self, num_classes: int = 100, model_name: str = 'efficientnet_b0', 
                 pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        
        if model_name == 'efficientnet_b0':
            backbone = torchvision.models.efficientnet_b0(weights=None)
            feat_dim = 1280
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        elif model_name == 'resnet50':
            backbone = torchvision.models.resnet50(weights=None)
            feat_dim = 2048
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        elif model_name == 'convnext_tiny':
            backbone = torchvision.models.convnext_tiny(weights=None)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.flatten(1)
        
        b, c = x.size()
        x_2d = x.view(b, c, 1, 1)
        x_2d = self.attention(x_2d)
        x = x_2d.flatten(1)
        
        x = self.classifier(x)
        return x

# ============================================================================
# ENSEMBLE METHODS
# ============================================================================

class EnsemblePredictor:
    """Ensemble prediction combining multiple models"""
    
    def __init__(self, model_paths: List[str], device: torch.device):
        """
        Args:
            model_paths: List of paths to trained model checkpoints
            device: Device to run models on
        """
        self.device = device
        self.models = []
        self.weights = []
        self.model_info = []
        
        print("Loading ensemble models...")
        for path in tqdm(model_paths):
            checkpoint = torch.load(path, map_location=device)
            
            # Create model
            model = CIFAR100Classifier(
                num_classes=100,
                model_name=checkpoint['architecture'],
                pretrained=False,
                dropout=checkpoint['dropout']
            ).to(device)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models.append(model)
            self.weights.append(checkpoint['val_acc'])
            self.model_info.append({
                'id': checkpoint['model_id'],
                'architecture': checkpoint['architecture'],
                'val_acc': checkpoint['val_acc']
            })
        
        # Normalize weights for weighted averaging
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
        
        print(f"Loaded {len(self.models)} models")
        for i, info in enumerate(self.model_info):
            print(f"  Model {info['id']}: {info['architecture']} "
                  f"(Val: {info['val_acc']:.2f}%, Weight: {self.weights[i]:.3f})")
    
    def predict_simple_average(self, x: torch.Tensor) -> torch.Tensor:
        """Simple averaging of predictions"""
        all_preds = []
        
        with torch.no_grad():
            for model in self.models:
                pred = F.softmax(model(x), dim=1)
                all_preds.append(pred)
        
        # Average predictions
        ensemble_pred = torch.stack(all_preds).mean(dim=0)
        return ensemble_pred
    
    def predict_weighted_average(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted averaging based on validation accuracy"""
        all_preds = []
        
        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                pred = F.softmax(model(x), dim=1)
                all_preds.append(pred * weight)
        
        # Weighted average
        ensemble_pred = torch.stack(all_preds).sum(dim=0)
        return ensemble_pred
    
    def predict_majority_voting(self, x: torch.Tensor) -> torch.Tensor:
        """Majority voting on predictions"""
        all_preds = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                _, predicted = pred.max(1)
                all_preds.append(predicted)
        
        # Stack predictions
        all_preds = torch.stack(all_preds)  # [num_models, batch_size]
        
        # Vote for each sample
        batch_size = x.size(0)
        final_preds = torch.zeros(batch_size, 100).to(self.device)
        
        for i in range(batch_size):
            votes = all_preds[:, i].cpu().numpy()
            vote_counts = Counter(votes)
            
            # Create probability distribution from votes
            for class_id, count in vote_counts.items():
                final_preds[i, class_id] = count / len(self.models)
        
        return final_preds
    
    def predict_rank_fusion(self, x: torch.Tensor) -> torch.Tensor:
        """Rank-based fusion of predictions"""
        all_preds = []
        
        with torch.no_grad():
            for model in self.models:
                pred = F.softmax(model(x), dim=1)
                all_preds.append(pred)
        
        # Convert to ranks
        batch_size, num_classes = all_preds[0].shape
        rank_scores = torch.zeros(batch_size, num_classes).to(self.device)
        
        for pred in all_preds:
            # Get ranks (higher prediction = lower rank number = better)
            ranks = pred.argsort(dim=1, descending=True).argsort(dim=1)
            # Convert to scores (lower rank = higher score)
            scores = num_classes - ranks.float()
            rank_scores += scores
        
        # Normalize
        rank_scores = F.softmax(rank_scores, dim=1)
        return rank_scores
    
    def predict_max_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Use prediction from most confident model"""
        all_preds = []
        all_confidences = []
        
        with torch.no_grad():
            for model in self.models:
                pred = F.softmax(model(x), dim=1)
                confidence, _ = pred.max(dim=1)
                all_preds.append(pred)
                all_confidences.append(confidence)
        
        # Stack predictions and confidences
        all_preds = torch.stack(all_preds)  # [num_models, batch_size, num_classes]
        all_confidences = torch.stack(all_confidences)  # [num_models, batch_size]
        
        # Select prediction from most confident model for each sample
        most_confident = all_confidences.argmax(dim=0)  # [batch_size]
        
        batch_size = x.size(0)
        final_preds = torch.zeros_like(all_preds[0])
        
        for i in range(batch_size):
            final_preds[i] = all_preds[most_confident[i], i]
        
        return final_preds

# ============================================================================
# EVALUATION
# ============================================================================

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

def evaluate_ensemble(ensemble: EnsemblePredictor, test_loader: DataLoader, 
                     device: torch.device, methods: List[str] = None) -> Dict:
    """Evaluate ensemble with different methods"""
    
    if methods is None:
        methods = ['simple_average', 'weighted_average', 'majority_voting', 
                  'rank_fusion', 'max_confidence']
    
    results = {}
    
    for method in methods:
        print(f"\nEvaluating with {method}...")
        
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'{method}')
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                
                # Get ensemble predictions
                if method == 'simple_average':
                    output = ensemble.predict_simple_average(data)
                elif method == 'weighted_average':
                    output = ensemble.predict_weighted_average(data)
                elif method == 'majority_voting':
                    output = ensemble.predict_majority_voting(data)
                elif method == 'rank_fusion':
                    output = ensemble.predict_rank_fusion(data)
                elif method == 'max_confidence':
                    output = ensemble.predict_max_confidence(data)
                
                # Compute accuracy
                acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
                correct_top1 += acc1.item() * target.size(0) / 100
                correct_top5 += acc5.item() * target.size(0) / 100
                total += target.size(0)
                
                pbar.set_postfix({
                    'top1': 100. * correct_top1 / total,
                    'top5': 100. * correct_top5 / total
                })
        
        top1_acc = 100. * correct_top1 / total
        top5_acc = 100. * correct_top5 / total
        
        results[method] = {
            'top1_acc': top1_acc,
            'top5_acc': top5_acc
        }
        
        print(f"{method}: Top-1 = {top1_acc:.2f}%, Top-5 = {top5_acc:.2f}%")
    
    return results

def plot_ensemble_comparison(results: Dict, individual_accs: List[float], 
                            save_path: str = 'ensemble_comparison.png'):
    """Plot comparison of ensemble methods"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top-1 accuracy comparison
    methods = list(results.keys())
    top1_accs = [results[m]['top1_acc'] for m in methods]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    bars = ax1.bar(range(len(methods)), top1_accs, color=colors)
    
    # Add best individual model line
    best_individual = max(individual_accs)
    ax1.axhline(y=best_individual, color='r', linestyle='--', linewidth=2,
                label=f'Best Single Model: {best_individual:.2f}%')
    
    # Add average individual model line
    avg_individual = np.mean(individual_accs)
    ax1.axhline(y=avg_individual, color='orange', linestyle='--', linewidth=2,
                label=f'Avg Single Model: {avg_individual:.2f}%')
    
    ax1.set_xlabel('Ensemble Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Ensemble Methods Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in methods], 
                        rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Improvement over best single model
    improvements = [acc - best_individual for acc in top1_accs]
    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars2 = ax2.bar(range(len(methods)), improvements, color=colors_imp)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
    
    ax2.set_xlabel('Ensemble Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement over Best Single (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Ensemble Improvement', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels([m.replace('_', ' ').title() for m in methods], 
                        rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars2, improvements):
        height = bar.get_height()
        va = 'bottom' if height > 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.2f}%', ha='center', va=va, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main ensemble prediction function"""
    
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
    
    # Load ensemble info
    if not os.path.exists('ensemble_models/ensemble_info.json'):
        print("ERROR: No ensemble found. Run cifar100_ensemble_trainer.py first!")
        return
    
    with open('ensemble_models/ensemble_info.json', 'r') as f:
        ensemble_info = json.load(f)
    
    # Get model paths
    model_paths = [model['model_path'] for model in ensemble_info['models']]
    individual_accs = [model['best_val_acc'] for model in ensemble_info['models']]
    
    print(f"\nFound {len(model_paths)} trained models")
    print(f"Best individual: {max(individual_accs):.2f}%")
    print(f"Average individual: {np.mean(individual_accs):.2f}%")
    
    # Create ensemble
    ensemble = EnsemblePredictor(model_paths, device)
    
    # Load test data
    print("\nLoading CIFAR-100 test set...")
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), 
                           (0.2675, 0.2565, 0.2761)),
    ])
    
    test_dataset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Evaluate ensemble
    print("\n" + "="*80)
    print("EVALUATING ENSEMBLE METHODS")
    print("="*80)
    
    results = evaluate_ensemble(ensemble, test_loader, device)
    
    # Print results
    print("\n" + "="*80)
    print("ENSEMBLE RESULTS")
    print("="*80)
    
    print(f"\nBest Single Model: {max(individual_accs):.2f}%")
    print(f"Average Single Model: {np.mean(individual_accs):.2f}%")
    print("\nEnsemble Methods:")
    print("-"*80)
    
    best_method = None
    best_acc = 0
    
    for method, metrics in results.items():
        improvement = metrics['top1_acc'] - max(individual_accs)
        print(f"{method.replace('_', ' ').title():20s}: "
              f"{metrics['top1_acc']:.2f}% (Top-5: {metrics['top5_acc']:.2f}%) "
              f"[{improvement:+.2f}%]")
        
        if metrics['top1_acc'] > best_acc:
            best_acc = metrics['top1_acc']
            best_method = method
    
    print("-"*80)
    print(f"\nBest Ensemble Method: {best_method.replace('_', ' ').title()}")
    print(f"Best Ensemble Accuracy: {best_acc:.2f}%")
    print(f"Improvement: +{best_acc - max(individual_accs):.2f}%")
    
    # Plot comparison
    plot_ensemble_comparison(results, individual_accs, 
                            'ensemble_models/ensemble_comparison.png')
    
    # Save results
    final_results = {
        'individual_models': {
            'best': max(individual_accs),
            'average': np.mean(individual_accs),
            'all': individual_accs
        },
        'ensemble_methods': results,
        'best_ensemble': {
            'method': best_method,
            'accuracy': best_acc,
            'improvement': best_acc - max(individual_accs)
        }
    }
    
    with open('ensemble_models/ensemble_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n✓ Results saved to ensemble_models/ensemble_results.json")
    print("✓ Comparison plot saved to ensemble_models/ensemble_comparison.png")

if __name__ == '__main__':
    print("="*80)
    print("CIFAR-100 ENSEMBLE PREDICTOR")
    print("="*80)
    main()
