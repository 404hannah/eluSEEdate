# 🚀 Mobile Deployment Quick Start Guide

**Goal**: Prepare ConvLSTM model for Android deployment in 3 phases

---

## ⚡ TL;DR - What We Need to Do

```bash
# Phase 1: Optimize model (MUST DO FIRST)
# 1. Modify model to use Global Average Pooling
# 2. Retrain the model from scratch
python train.py

# Phase 2: Convert to mobile format (automated)
pip install onnx onnxruntime onnx-tf tensorflow
python mobile_deploy.py --model best_convlstm.pth --mode all

# Phase 3: Verify results
# Check that accuracy is within 5% and model is < 2 MB
```

---

## 📋 Phase 1: Model Optimization (Critical!)

### The Problem
The current model has a **524,288-parameter FC layer** that makes it too large for mobile (7.4 MB).

### The Solution: Global Average Pooling
Replace the massive FC layer with a tiny one using Global Average Pooling.

### Steps:

**Modify Existing Model Manually**
1. Open `models/conv_lstm_classifier.py`
2. In `ConvLSTMModel.__init__()`, add:
   ```python
   # Add after self.convlstm definition:
   self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
   
   # Change this line:
   # OLD: self.fc = nn.Linear(hidden_dim[-1] * HEIGHT * WIDTH, num_classes)
   # NEW: self.fc = nn.Linear(hidden_dim[-1], num_classes)
   ```
3. In `ConvLSTMModel.forward()`, modify:
   ```python
   # After: h_last = layer_output_list[0][:, -1, :, :, :]
   
   # Add these lines:
   h_last = self.global_avg_pool(h_last)  # NEW: Apply GAP
   h_last = h_last.view(h_last.size(0), -1)  # Flatten
   
   # Remove old flatten line:
   # DELETE: h_last = h_last.reshape(h_last.size(0), -1)
   ```
4. Run: `python train.py`

### Expected Results:
- ✅ Model size: 7.4 MB → 1.5 MB (-80%)
- ✅ Parameters: 1.57M → ~150K (-90%)
- ⚠️ Accuracy: May drop 2-5% (acceptable)

### Test the Optimized Model:
```bash
# Compare original vs optimized
cd models
python conv_lstm_classifier_optimized.py
```

---

## 📋 Phase 2: Format Conversion (Automated)

### Install Dependencies:
```bash
pip install onnx onnxruntime
pip install onnx-tf tensorflow
pip install netron  # Optional: for visualization
```

### Run Conversion Pipeline:
```bash
# Full automated pipeline (recommended)
python mobile_deploy.py --model best_convlstm.pth --mode all

# OR step-by-step:
python mobile_deploy.py --mode export    # PyTorch → ONNX
python mobile_deploy.py --mode convert   # ONNX → TFLite
python mobile_deploy.py --mode benchmark # Test performance
```

### What This Does:
1. **PyTorch → ONNX**: Creates portable `convlstm_model.onnx`
2. **ONNX → TensorFlow**: Creates `convlstm_tf_model/` directory
3. **TensorFlow → TFLite Float32**: Creates `convlstm_model_float32.tflite` (~1.5 MB)
4. **TFLite → INT8 Quantized**: Creates `convlstm_model_int8.tflite` (~500-800 KB)

### Visualize Model (Optional):
```bash
netron convlstm_model.onnx
# Opens interactive model viewer in browser
```

---

## 📋 Phase 3: Validation

### Check Model Sizes:
```bash
python mobile_deploy.py --mode benchmark
```

**Success Criteria:**
- ✅ TFLite INT8 model < 2 MB (ideally < 1 MB)
- ✅ Inference latency < 300 ms on CPU
- ✅ Accuracy within 5% of original

### Test Accuracy:
```python
# Add to tester.py or create new script
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='mobile_models/convlstm_model_int8.tflite')
interpreter.allocate_tensors()

# Test on test dataset
# Compare accuracy with PyTorch model
# Target: >= (pytorch_accuracy - 5%)
```

---

## 📦 Deliverables for Android Team

Once everything passes, deliver:

1. **Model File**: `convlstm_model_int8.tflite`
2. **Input Specifications**:
   - Shape: `[1, 30, 6, 128, 128]`
   - Type: `float32`
   - Range: `[0, 1]` (normalized)
3. **Output Specifications**:
   - Shape: `[1, 3]`
   - Type: `float32`
   - Classes: `0=Front, 1=Left, 2=Right`
4. **Preprocessing Steps**:
   ```
   1. Extract 30 frames from 3-second video at 10 FPS
   2. Resize each frame to 128×128 pixels
   3. Normalize pixel values: [0, 255] → [0, 1]
   4. Stack frames: [30, 128, 128, 3]
   5. Add intent channels (if used): [30, 128, 128, 6]
   6. Reshape to model input: [1, 30, 6, 128, 128]
   ```
5. **Performance Metrics**:
   - Accuracy: ____%
   - Latency (CPU): ___ ms
   - Model size: ___ KB

---

## ✅ Success Checklist

Before considering deployment complete:

- [ ] Model retrained with Global Average Pooling
- [ ] Training accuracy >= 85% (or comparable to original)
- [ ] Test accuracy within 5% of original model
- [ ] ONNX export successful and validated
- [ ] TFLite conversion successful
- [ ] INT8 quantization applied
- [ ] Model size < 2 MB (preferably < 1 MB)
- [ ] CPU inference latency < 300 ms
- [ ] Tested on sample videos with correct predictions
- [ ] Documentation prepared for Android team
- [ ] Model artifacts organized in `mobile_models/` directory

---

## 🆘 Need Help?

1. Read [MOBILE_DEPLOYMENT.md](MOBILE_DEPLOYMENT.md) for detailed explanations
2. Check troubleshooting sections in documentation
3. Verify PyTorch, ONNX, and TensorFlow versions are compatible
4. Test each phase independently before running full pipeline

---

**Last Updated**: February 20, 2026  
**Version**: 1.0  