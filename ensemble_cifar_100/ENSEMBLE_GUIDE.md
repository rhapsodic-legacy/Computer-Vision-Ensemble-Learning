# CIFAR-100 Ensemble Guide

## 🎯 Goal: Push from 70-72% to 74-77%+ with Ensembling

Congratulations on achieving 70-72% with the enhanced model! Now let's use ensemble methods to squeeze out the last few percentage points.

## 📊 What is Ensembling?

**Core Idea:** Combine predictions from multiple models to get better results than any single model.

**Why it works:**
- Different models make different mistakes
- Averaging reduces variance
- Diverse models capture different patterns
- "Wisdom of the crowd" effect

**Expected gain:** +2-5% accuracy

## 🔥 Ensemble Strategies

### 1. **Diversity is Key**

Train models that are different from each other:

```
Model 1: EfficientNet-B0, seed=42, mixup=0.2
Model 2: ResNet50, seed=123, mixup=0.1
Model 3: ConvNeXt-Tiny, seed=456, mixup=0.3
Model 4: EfficientNet-B0, seed=789, dropout=0.4
Model 5: ResNet50, seed=2024, dropout=0.2
```

**Sources of diversity:**
- ✅ Different architectures (EfficientNet, ResNet, ConvNeXt)
- ✅ Different random seeds (different weight initialization)
- ✅ Different hyperparameters (mixup, dropout, LR)
- ✅ Different training data (optional: different augmentations)

### 2. **Number of Models**

**Diminishing returns:**
```
1 model  → 70.0% (baseline)
2 models → 71.5% (+1.5%)
3 models → 72.5% (+2.5%)
5 models → 73.5% (+3.5%)
7 models → 74.0% (+4.0%)
10 models → 74.3% (+4.3%)
```

**Sweet spot:** 5-7 models
- Good accuracy improvement
- Manageable training time
- Not too much inference overhead

### 3. **Ensemble Methods**

We've implemented 5 different methods:

#### A. Simple Averaging (Most Common)
```python
prediction = mean([model1(x), model2(x), model3(x), ...])
```
- Pros: Simple, robust, usually works well
- Cons: Treats all models equally
- **Expected: +2-3%**

#### B. Weighted Averaging (Better)
```python
prediction = Σ(weight_i * model_i(x))
# where weight_i = val_acc_i / Σ(val_acc)
```
- Pros: Better models have more influence
- Cons: Requires validation set
- **Expected: +2.5-4%**

#### C. Majority Voting
```python
prediction = most_common([argmax(model1(x)), argmax(model2(x)), ...])
```
- Pros: Robust to outliers
- Cons: Loses probability information
- **Expected: +2-3%**

#### D. Rank Fusion
```python
# Combine based on ranking of predictions
for each model:
    ranks = argsort(predictions)
    scores = 100 - ranks
ensemble_score = sum(scores across models)
```
- Pros: Robust to scale differences
- Cons: More complex
- **Expected: +2.5-3.5%**

#### E. Max Confidence
```python
# Use prediction from most confident model
confidences = [max(model_i(x)) for all models]
prediction = model_with_highest_confidence(x)
```
- Pros: Leverages model certainty
- Cons: Might amplify errors
- **Expected: +1.5-2.5%**

## 📁 Files Provided

### 1. `cifar100_ensemble_trainer.py`
**Purpose:** Train multiple diverse models

**What it does:**
- Trains 5 models with different configurations
- Saves each model's checkpoint
- Tracks individual model performance
- Creates ensemble metadata

**Configuration:**
```python
class EnsembleConfig:
    NUM_MODELS = 5  # How many models to train
    SELECTION_STRATEGY = 'diverse'  # or 'best'
    
    # Diversity sources
    ARCHITECTURES = ['efficientnet_b0', 'resnet50', 'convnext_tiny']
    RANDOM_SEEDS = [42, 123, 456, 789, 2024]
    MIXUP_ALPHAS = [0.1, 0.2, 0.3]
    DROPOUTS = [0.2, 0.3, 0.4]
```

**Training time:** ~3.5 hours per model × 5 = **17.5 hours** (GPU)

### 2. `cifar100_ensemble_predictor.py`
**Purpose:** Combine trained models for predictions

**What it does:**
- Loads all trained ensemble models
- Evaluates 5 different ensemble methods
- Compares to best single model
- Creates visualization
- Saves results

**Output:**
- Individual model accuracies
- Ensemble accuracies (all methods)
- Best ensemble method
- Improvement over single model
- Comparison plot

## 🚀 Quick Start

### Step 1: Train Ensemble (17.5 hours)

```bash
# Install dependencies (if needed)
pip install torch torchvision matplotlib tqdm --break-system-packages

# Train 5 diverse models
python cifar100_ensemble_trainer.py

# This will create:
# - ensemble_models/model_1_best.pth
# - ensemble_models/model_2_best.pth
# - ensemble_models/model_3_best.pth
# - ensemble_models/model_4_best.pth
# - ensemble_models/model_5_best.pth
# - ensemble_models/ensemble_info.json
```

**What to expect during training:**
```
================================================================================
Training Model #1
Architecture: efficientnet_b0 | Seed: 42
MixUp: 0.2 | Dropout: 0.3 | LR: 0.001
================================================================================

Epoch  20/200 | Val Acc: 62.34% | Top-5: 87.12% | LR: 0.000856
Epoch  40/200 | Val Acc: 67.89% | Top-5: 89.45% | LR: 0.000714
Epoch  60/200 | Val Acc: 69.23% | Top-5: 90.12% | LR: 0.000572
...
Epoch 180/200 | Val Acc: 71.45% | Top-5: 91.23% | LR: 0.000028

✓ Model #1 complete | Best Val Acc: 71.45%

[Repeats for models 2-5...]
```

### Step 2: Combine Predictions (5 minutes)

```bash
# Evaluate ensemble
python cifar100_ensemble_predictor.py

# This will:
# 1. Load all 5 models
# 2. Test 5 ensemble methods
# 3. Show results and create plots
```

**Expected output:**
```
================================================================================
ENSEMBLE RESULTS
================================================================================

Best Single Model: 71.45%
Average Single Model: 70.23%

Ensemble Methods:
--------------------------------------------------------------------------------
Simple Average      : 73.12% (Top-5: 92.34%) [+1.67%]
Weighted Average    : 73.89% (Top-5: 92.56%) [+2.44%]
Majority Voting     : 72.78% (Top-5: 92.01%) [+1.33%]
Rank Fusion         : 73.45% (Top-5: 92.23%) [+2.00%]
Max Confidence      : 72.34% (Top-5: 91.78%) [+0.89%]
--------------------------------------------------------------------------------

Best Ensemble Method: Weighted Average
Best Ensemble Accuracy: 73.89%
Improvement: +2.44%
```

## 📈 Expected Results Breakdown

### Starting Point (Your Current Single Model)
```
Top-1 Accuracy: 70-72%
Top-5 Accuracy: 91-92%
```

### After Ensemble Training

**Individual Models:**
```
Model 1 (EfficientNet): 71.5%
Model 2 (ResNet50):     70.8%
Model 3 (ConvNeXt):     71.2%
Model 4 (EfficientNet): 70.9%
Model 5 (ResNet50):     71.0%

Best Single: 71.5%
Average:     71.1%
```

**Ensemble Results:**
```
Simple Average:    73.1% (+1.6% over best single)
Weighted Average:  73.9% (+2.4% over best single) ← Best!
Majority Voting:   72.8% (+1.3% over best single)
Rank Fusion:       73.4% (+1.9% over best single)
Max Confidence:    72.3% (+0.8% over best single)
```

### Final Target
```
🎯 Target:  74-77% top-1 accuracy
🎯 Top-5:   92-94%
🎯 Status:  Achievable with 5-7 model ensemble!
```

## 💡 Advanced Tips

### 1. **Quick Ensemble (3 models, faster)**

If you're time-constrained:

```python
# In cifar100_ensemble_trainer.py, change:
class EnsembleConfig:
    NUM_MODELS = 3  # Instead of 5
```

Expected: +1.5-2.5% (instead of +2-4%)
Time: ~10 hours instead of 17.5 hours

### 2. **Mega Ensemble (10 models, maximum accuracy)**

For maximum performance:

```python
class EnsembleConfig:
    NUM_MODELS = 10
    RANDOM_SEEDS = [42, 123, 456, 789, 2024, 3141, 5926, 5358, 9793, 2384]
```

Expected: +3-5%
Time: ~35 hours

### 3. **Mix Best Configurations**

Instead of 'diverse' strategy, use 'best':

```python
class EnsembleConfig:
    SELECTION_STRATEGY = 'best'  # All use best hyperparams, different seeds
```

- Pros: Each model is strong
- Cons: Less diversity
- Expected: Similar or slightly better results

### 4. **Test-Time Augmentation + Ensemble**

Combine TTA with ensemble:

```python
# In predictor, for each model:
predictions = []
for _ in range(5):  # 5 TTA crops
    augmented = apply_tta(image)
    predictions.append(model(augmented))
model_pred = average(predictions)

# Then ensemble across models
ensemble_pred = weighted_average([model1_pred, model2_pred, ...])
```

Expected: Additional +0.5-1% (total: +3-5%)

### 5. **Progressive Ensemble**

Train models progressively:
1. Train model 1 → 71%
2. Train model 2 → 70.5%, ensemble → 72%
3. Train model 3 → 71%, ensemble → 72.5%
4. Continue until improvement plateaus

Advantage: Can stop early if satisfied

## 🎓 Understanding the Gain

### Why +2-5% and not +10%?

**Law of Diminishing Returns:**
- Models learn similar patterns
- Easy samples: all models get right
- Hard samples: all models get wrong
- Medium samples: ensemble helps!

**Breakdown:**
```
Easy samples (60%):   All models correct
Medium samples (25%): Ensemble helps (+10-20% gain)
Hard samples (15%):   All models incorrect

Overall gain = 25% × 15% average boost = ~4% improvement
```

### What About More Models?

```
Models  Improvement  Reason
──────────────────────────────────────
1       0%          Baseline
2       +1.5%       Big diversity gain
3       +2.5%       Good diversity
5       +3.5%       Diminishing returns
7       +4.0%       Smaller gains
10      +4.5%       Marginal improvement
20      +5.0%       Not worth it
```

**Sweet spot:** 5-7 models for best effort/reward ratio

## 🏆 Achieving 75%+

To break 75%, you'd need:

### Strategy 1: Ensemble + TTA
```
Single model:       71%
Ensemble (5):       +3% → 74%
TTA:               +1% → 75%
```

### Strategy 2: Better Base Models + Ensemble
```
Improved singles:   72% (longer training, better HP)
Ensemble (5):       +2.5% → 74.5%
TTA:               +0.5% → 75%
```

### Strategy 3: Mega Ensemble
```
Single model:       71%
Ensemble (10):      +4% → 75%
```

## 📊 Tracking Your Progress

### Ensemble Training Logs

During training, monitor:
```
Individual Model Performance:
Model 1: 71.5% ✓ (Good)
Model 2: 70.8% ✓ (Good)
Model 3: 68.9% ✗ (Weak - might retrain)
Model 4: 71.2% ✓ (Good)
Model 5: 70.5% ✓ (Good)
```

**Red flags:**
- Model significantly worse than others (>2% gap)
- Model didn't converge (early stopping too early)
- Model identical to another (no diversity)

**Solutions:**
- Retrain weak models with different hyperparameters
- Increase training epochs
- Change architecture or seed

### Ensemble Prediction Results

Look for:
```
✅ Ensemble > Best Single (2%+)
✅ Multiple methods > 73%
✅ Weighted average is best
✅ Low variance across methods
```

**Warning signs:**
- Ensemble barely better than single model
- High variance across methods
- Voting methods outperform averaging (suggests poor probability calibration)

## 🔧 Troubleshooting

### Problem 1: Training Taking Too Long
```python
# Solution 1: Reduce epochs
config.NUM_EPOCHS = 150  # Instead of 200

# Solution 2: Train fewer models
config.NUM_MODELS = 3  # Instead of 5

# Solution 3: Use smaller models only
config.ARCHITECTURES = ['efficientnet_b0']  # Remove resnet50, convnext
```

### Problem 2: Ensemble Not Improving Much
```python
# Solution 1: Increase diversity
# Use more different architectures
config.ARCHITECTURES = ['efficientnet_b0', 'resnet50', 'convnext_tiny', 
                       'mobilenet_v3_large', 'regnet_y_400mf']

# Solution 2: Vary hyperparameters more
config.MIXUP_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
config.DROPOUTS = [0.1, 0.2, 0.3, 0.4, 0.5]

# Solution 3: Use different augmentation strategies per model
```

### Problem 3: Out of Memory
```python
# Solution: Load models one at a time during prediction
# Modify predictor to use iterative approach instead of loading all at once
```

## 🎯 Expected Timeline

### Week 1: Train Ensemble
- Day 1-2: Train models 1-2 (7 hours)
- Day 3-4: Train models 3-4 (7 hours)
- Day 5: Train model 5 (3.5 hours)
- Day 6-7: Run ensemble predictor, analyze results

### Week 2: Optimize
- Retrain any weak models
- Experiment with different ensemble methods
- Add TTA if needed
- Reach 74-75%+

## 📋 Checklist

Before starting:
- [ ] Current single model achieves 70-72%
- [ ] Have 18+ hours for training (or reduce NUM_MODELS)
- [ ] Installed all dependencies
- [ ] Have GPU/MPS available (recommended)

During training:
- [ ] Monitor individual model performance
- [ ] Check for diversity (models should differ by 1-2%)
- [ ] Ensure no crashes/errors
- [ ] Save checkpoints regularly

After training:
- [ ] Run ensemble predictor
- [ ] Compare all 5 ensemble methods
- [ ] Verify improvement > 2%
- [ ] Save best ensemble configuration

## 🎉 Success Criteria

You'll know you've succeeded when:
- ✅ Best ensemble method > 73.5%
- ✅ Improvement over single model > 2%
- ✅ Top-5 accuracy > 92%
- ✅ Multiple ensemble methods work well
- ✅ Results are reproducible

## 📚 Next Steps After 75%

If you reach 75% and want even more:

1. **Knowledge Distillation** (+1-2%)
   - Use ensemble as teacher
   - Train single student model
   - Student can match ensemble!

2. **Semi-Supervised Learning** (+2-3%)
   - Use unlabeled data
   - Pseudo-labeling
   - Consistency regularization

3. **Neural Architecture Search** (+1-2%)
   - Find optimal architecture
   - AutoML approaches

4. **Extended Training** (+0.5-1%)
   - Train 400-500 epochs
   - Very low learning rate

But honestly, 74-75% is **excellent** for CIFAR-100! 🎉

---

**Good luck with your ensemble! You're about to join the 75% club!** 🚀
