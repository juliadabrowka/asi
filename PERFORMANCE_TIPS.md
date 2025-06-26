# Performance Optimization Tips

## 🚀 Quick Training Options

### Option 1: Quick Training Script (Recommended for Development)
```bash
python quick_train.py
```
- **Time**: ~30 seconds
- **Quality**: Good for development/testing
- **Best for**: Quick iterations and testing

### Option 2: Fast Kedro Pipeline
```bash
kedro run --pipeline model_training --env local
```
- **Time**: ~30 seconds
- **Quality**: Good for development
- **Uses**: `optimize_for_deployment` preset

### Option 3: Balanced Kedro Pipeline
```bash
kedro run --pipeline model_training
```
- **Time**: ~2 minutes
- **Quality**: Better for production
- **Uses**: `medium_quality` preset

## ⚡ Performance Optimizations Applied

### 1. Training Time Reduction
- **Before**: 600 seconds (10 minutes)
- **After**: 30-120 seconds (0.5-2 minutes)

### 2. Preset Optimization
- **Before**: `best_quality` (slowest, highest quality)
- **After**: `optimize_for_deployment` or `medium_quality` (faster, still good quality)

### 3. Resource Management
- Limited CPU usage to prevent system slowdown
- Added GPU support when available
- Reduced verbosity for less overhead

### 4. Inference Optimization
- Model warm-up on load
- Faster prediction methods
- Optimized data handling

## 🔧 Configuration Files

### Fast Development (`conf/local/parameters.yml`)
```yaml
time_limit: 30
presets: optimize_for_deployment
epochs: 20
```

### Balanced Production (`conf/base/parameters.yml`)
```yaml
time_limit: 120
presets: medium_quality
epochs: 50
```

## 💡 Additional Tips

### For Even Faster Training:
1. **Reduce dataset size** for testing
2. **Use fewer features** if possible
3. **Enable GPU** if available (set `CUDA_VISIBLE_DEVICES`)

### For Better Quality:
1. **Increase time_limit** to 300+ seconds
2. **Use `best_quality` preset**
3. **Increase epochs** to 100+

### System Optimization:
1. **Close other applications** during training
2. **Use SSD storage** for faster I/O
3. **Ensure sufficient RAM** (8GB+ recommended)

## 🐛 Troubleshooting

### If training is still slow:
1. Check CPU usage with Task Manager
2. Ensure no other ML processes are running
3. Try the quick training script first
4. Consider reducing dataset size for testing

### If model quality is poor:
1. Increase `time_limit` in parameters
2. Use `medium_quality` or `best_quality` preset
3. Check data quality and preprocessing
4. Ensure sufficient training data 