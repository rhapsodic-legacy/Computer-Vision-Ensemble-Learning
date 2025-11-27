# ENSEMBLE QUICK REFERENCE

## 🎯 Goal
Push from **70-72%** → **74-77%** using ensemble methods

## ⚡ Quick Commands

### Step 1: Train Ensemble (One Command!)
```bash
python cifar100_ensemble_trainer.py
```
- Trains 5 diverse models
- Takes ~17.5 hours (GPU)
- Saves to `ensemble_models/`

### Step 2: Get Predictions (5 minutes)
```bash
python cifar100_ensemble_predictor.py
```
- Loads all 5 models
- Tests 5 ensemble methods
- Shows best method
- Creates comparison plot

## 📊 What to Expect

### During Training
```
Model 1: EfficientNet-B0 → 71.5%
Model 2: ResNet50       → 70.8%
Model 3: ConvNeXt-Tiny  → 71.2%
Model 4: EfficientNet-B0 → 70.9%
Model 5: ResNet50       → 71.0%
```

### After Ensemble
```
Best Single Model:     71.5%
Weighted Average:      73.9%  (+2.4%) ✓
Simple Average:        73.1%  (+1.6%)
Rank Fusion:          73.4%  (+1.9%)
```

## 🎛️ Quick Tweaks

### Faster Training (3 models, 10 hours)
```python
# In cifar100_ensemble_trainer.py:
class EnsembleConfig:
    NUM_MODELS = 3  # Change from 5
```

### Maximum Accuracy (10 models, 35 hours)
```python
class EnsembleConfig:
    NUM_MODELS = 10  # Change from 5
```

### Use Only Best Config (less diversity)
```python
class EnsembleConfig:
    SELECTION_STRATEGY = 'best'  # Change from 'diverse'
```

## 📈 Expected Gains

| # Models | Training Time | Improvement | Final Acc |
|----------|---------------|-------------|-----------|
| 1        | 3.5h         | 0%          | 71%       |
| 2        | 7h           | +1.5%       | 72.5%     |
| 3        | 10.5h        | +2%         | 73%       |
| 5        | 17.5h        | +2.5-3.5%   | 74%       |
| 10       | 35h          | +3.5-4.5%   | 75%+      |

## 🔍 Files Created

After running both scripts:
```
ensemble_models/
├── model_1_best.pth          # Model 1 checkpoint
├── model_2_best.pth          # Model 2 checkpoint
├── model_3_best.pth          # Model 3 checkpoint
├── model_4_best.pth          # Model 4 checkpoint
├── model_5_best.pth          # Model 5 checkpoint
├── ensemble_info.json        # Training metadata
├── ensemble_results.json     # Prediction results
└── ensemble_comparison.png   # Visualization
```

## 🎯 Success Metrics

✅ Individual models: 70-72% each
✅ Best ensemble: 73.5-75%
✅ Improvement: +2-4%
✅ Top-5 accuracy: 92-94%

## 🐛 Common Issues

### Out of Memory
```python
# Reduce batch size
config.BATCH_SIZE = 64  # Instead of 128
```

### Taking Too Long
```python
# Reduce epochs
config.NUM_EPOCHS = 150  # Instead of 200

# OR train fewer models
config.NUM_MODELS = 3  # Instead of 5
```

### Models Too Similar
```python
# Increase diversity
config.MIXUP_ALPHAS = [0.0, 0.2, 0.4]  # More variation
config.DROPOUTS = [0.1, 0.3, 0.5]      # More variation
```

## 💡 Pro Tips

1. **Check individual models first**
   - All should be 69-72%
   - If one is <68%, retrain it

2. **Weighted average usually wins**
   - It accounts for model quality
   - Typically 0.5-1% better than simple average

3. **Don't need perfect models**
   - 70-72% individuals are fine
   - Diversity > perfection

4. **Monitor training**
   - Check every 50 epochs
   - Early stopping is OK
   - Save best checkpoints

## 🚀 Roadmap to 75%

```
Current:         70-72% (single model)
↓
Train Ensemble:  +2.5% (weighted average)
↓
Add TTA:        +0.5-1% (test-time augmentation)
↓
Final:          74-75%! 🎉
```

## 📞 Next Steps

1. Run `cifar100_ensemble_trainer.py`
2. Wait ~17.5 hours (or go do something else!)
3. Run `cifar100_ensemble_predictor.py`
4. Check if best ensemble > 73.5%
5. If not, retrain weak models or add more models
6. Celebrate when you hit 74%+ ! 🎊

---

**That's it! Two commands to 74%+ accuracy!** ⚡
