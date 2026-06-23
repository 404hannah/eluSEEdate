# 📱 Mobile Deployment Guide for ConvLSTM Model

**Target Platform**: Android Mobile Devices  
**Objective**: Optimize and deploy ConvLSTM turn prediction model for real-time inference on resource-constrained devices

---

## 📋 Table of Contents

- [Overview](#overview)
- [Deployment Pipeline](#deployment-pipeline)
- [Phase 1: Model Optimization](#phase-1-model-optimization)
- [Phase 2: Model Conversion](#phase-2-model-conversion)
- [Phase 3: Validation & Benchmarking](#phase-3-validation--benchmarking)
- [Performance Targets](#performance-targets)
- [Deployment Checklist](#deployment-checklist)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

### Current Model Specifications

| Metric             | Value                     | Status                |
| ------------------ | ------------------------- | --------------------- |
| **Model Size**     | ~7.4 MB                   | ❌ Too large          |
| **Inference Time** | ~200-300ms (GPU)          | ⚠️ Untested on mobile |
| **Input Shape**    | [1, 30, 6, 128, 128]      | ✅ Acceptable         |
| **Output Shape**   | [1, 3]                    | ✅ Optimal            |
| **Bottleneck**     | FC layer (524,288 params) | ❌ Critical issue     |

### Mobile Deployment Challenges

1. **Memory Footprint**: 524,288-parameter FC layer causes excessive memory usage
2. **Model Size**: 7.4 MB exceeds recommended 5 MB for mobile apps
3. **Inference Latency**: GPU-optimized model may be slow on mobile CPU/NPU
4. **Format Compatibility**: PyTorch `.pth` not directly usable on Android

### Solution Strategy

```
Original Model (7.4 MB)
    ↓
[PHASE 1] Model Optimization
    ├── Global Average Pooling (7.4 MB → 1.5 MB)
    ├── Knowledge Distillation (optional)
    └── Pruning (optional)
    ↓
Optimized Model (1.5-2 MB)
    ↓
[PHASE 2] Format Conversion
    ├── PyTorch → ONNX
    ├── ONNX → TensorFlow
    └── TensorFlow → TFLite + INT8 Quantization
    ↓
Mobile-Ready Model (< 1 MB)
    ↓
[PHASE 3] Validation & Deployment
    ├── Accuracy verification
    ├── Latency benchmarking
    └── Android integration testing
```

---

## 🔄 Deployment Pipeline

### Prerequisites

Install required packages: torch, torchvision, onnx, onnxruntime, tensorflow, tensorflow-lite, onnx2tf (onnx-tf is outdated), and netron (model visualization tool).

---

## 🎯 Phase 1: Model Optimization

### 1.1 Add Global Average Pooling (Critical)

**Objective**: Replace massive FC layer with efficient pooling operation

**Current Architecture Bottleneck**:
The last ConvLSTM hidden state has shape [B, 32, 128, 128], which gets flattened to [B, 524288] before passing through a fully connected layer with 524288 → 3 parameters (1,572,867 parameters total).

**Optimized Architecture**:
Using Global Average Pooling, the hidden state [B, 32, 128, 128] is reduced to [B, 32, 1, 1] through adaptive average pooling, then flattened to [B, 32]. The FC layer becomes 32 → 3 with only 99 parameters.

**Implementation Steps**:
In the file models/conv_lstm_classifier.py, modify the ConvLSTMModel class:

- In the `__init__()` method: Replace the large FC layer with an AdaptiveAvgPool2d layer (output size 1×1) followed by a smaller FC layer (hidden_dim[-1] → num_classes)
- In the `forward()` method: Replace the reshape operation with global average pooling before flattening to [B, 32]

**Expected Impact**:

- **Parameter Reduction**: 1,572,867 → 99 params (-99.99%)
- **Model Size**: 7.4 MB → ~1.5 MB (-80%)
- **Inference Speed**: 10-20% faster (less computation)
- **Accuracy Impact**: Expect 2-5% drop (acceptable)

**Validation**:

1. Modify model architecture as described above
2. Retrain model from scratch using train.py
3. Compare accuracy using tester.py (Target: Original Accuracy - 5% or better)

---

### 1.2 Model Pruning (Optional)[Not needed now]

**Objective**: Remove redundant connections in ConvLSTM layers

**Strategy**: L1 Unstructured Pruning
Remove 30% of the least important weights from ConvLSTM layers using L1 unstructured pruning. Iterate through Conv2d modules, apply L1 pruning to 30% of weights, then make the pruning permanent. After training, load the model, apply pruning, and save the pruned model.

**Expected Impact**:

- **Size Reduction**: Additional 20-30%
- **Speed Improvement**: 5-10% faster
- **Accuracy Impact**: < 1% drop

**When to Use**: If model is still > 2 MB after GAP implementation

---

### 1.3 Knowledge Distillation (Advanced, Optional)[Not needed now]

**Objective**: Train smaller "student" model to mimic larger "teacher"

**Strategy**:
Use a Teacher-Student approach where the Teacher is the original ConvLSTM with [64, 32] hidden dims and the Student is a smaller ConvLSTM with [32, 16] hidden dims. Implement a combined distillation loss function where the student learns both from the teacher's soft predictions (using KL divergence with temperature scaling) and from the true labels (using cross-entropy). Use alpha parameter (e.g., 0.7) to weight the distillation loss versus classification loss, and temperature (e.g., 3.0) to soften the probability distributions.

**Expected Impact**:

- **Size Reduction**: 50-60% smaller model
- **Accuracy**: Within 3-5% of teacher model
- **Training Time**: 2x longer

**When to Use**: If inference latency is still too high after other optimizations

---

## 🔧 Phase 2: Model Conversion

### 2.1 PyTorch → ONNX (Universal Format)

**Objective**: Convert PyTorch model to portable ONNX format

**Implementation**:
Load the trained PyTorch model, set it to evaluation mode, create a dummy input tensor matching the input shape [1, 20, 6, 128, 128], then use torch.onnx.export() with the following key settings:

- opset_version=12 for better compatibility (not successful when implemented)
- do_constant_folding=True to optimize constants (always enabled)
- Define input/output names ('video_input', 'direction_output')
- Set dynamic_axes for batch_size flexibility (error during conversion to TensorFlow)
  After export, verify the ONNX model using onnx.checker.check_model() to ensure validity.

**Validation**:
Load the ONNX model using onnxruntime, create a test input with shape [1, 20, 6, 128, 128], run inference, and verify the output shape is (1, 3).

**Visualization**:
Use Netron to visualize the model architecture by opening the .onnx file in the Netron application.

---

### 2.2 ONNX → TensorFlow (Intermediate Step)

**Objective**: Convert ONNX to TensorFlow format (required for TFLite)

**Implementation**:
Use onnx2tf.convert() function to convert the ONNX model to TensorFlow and TFLite models.

**Note**: Some ONNX operations may not be fully supported. If conversion fails, consider direct PyTorch → TensorFlow conversion (not recommended) using libraries like pytorch2keras.

---

### 2.3 TensorFlow → TFLite + Quantization (Final Format)

**Objective**: Convert to TensorFlow Lite with INT8 quantization for mobile

**Implementation**:
Create a TFLiteConverter from the SavedModel directory.

**Post-Training Quantization**

- **Integer Quantized Model** <br>
  For INT8 quantization, enable optimizations and provide a representative dataset (generator function yielding 100-500 sample inputs). Configure full integer quantization with INT8 input/output types for best performance, or use dynamic range quantization (weights only) if no representative dataset is available. Convert the model and save to a .tflite file. <br>
  The representative dataset should be created from actual training/validation data, iterating through samples and yielding properly formatted numpy arrays with shape [1, 30, 6, 128, 128].

- **Dynamic Range Quantized Model** <br>
  The weights of this type uses 8-bit precision. The model size decreases by 400% from insignificant change in speed and accuracy.

**Comparison between Integer and Dynamic Range Quantized Models**

| Model         | Accuracy | Size  | Speed |
| ------------- | -------- | ----- | ----- |
| Base Model    | 98.53%   | 55.08 | 1.04  |
| Integer       | 92.82%   | 5.22  | 0.4   |
| Dynamic Range | 98.15%   | 5.33  | 0.3   |

Source: LearnOpenCV (2022)
https://learnopencv.com/tensorflow-lite-model-optimization-for-on-device-machine-learning/

Conclusion: **Dynamic Range**

**Expected Sizes**:

- **Float32 TFLite**: ~1.5 MB
- **INT8 Quantized TFLite**: ~400-800 KB
- **Dynamic Range**: ~800 KB - 1 MB

**Validation**:
Load the TFLite model using tf.lite.Interpreter, allocate tensors, retrieve input/output details, then run inference with a test input. Verify the input shape [1, 20, 6, 128, 128] and output shape (1, 3), along with their data types (float32 or int8 depending on quantization).

---

## ✅ Phase 3: Validation & Benchmarking

### 3.1 Accuracy Validation

**Objective**: Verify model accuracy is maintained throughout optimization pipeline

**Pipeline Testing**:
Create a comparison function that runs inference on the same test dataset across all three formats (PyTorch, ONNX, TFLite). Load each model, iterate through the test_loader, collect predictions from all formats, and calculate accuracy metrics using sklearn. Compare results to ensure:

- ONNX accuracy is within 0.5% of PyTorch (numerical precision differences)
- TFLite Float32 is within 1% of PyTorch
- TFLite INT8 is within 2-5% of PyTorch (acceptable quantization loss)

Generate a classification report for each format showing precision, recall, and F1-score for all three classes (Front, Left, Right).

**Acceptance Criteria**:

- **ONNX Accuracy**: Within 0.5% of PyTorch (numerical precision differences)
- **TFLite Float32**: Within 1% of PyTorch
- **TFLite INT8**: Within 2-5% of PyTorch (acceptable quantization loss)

---

### 3.2 Latency Benchmarking

**Objective**: Measure inference speed on mobile-like CPU environment

**Implementation**:
Create a benchmarking function that loads the TFLite model, performs warmup runs (10 iterations) to stabilize performance, then measures latency over 100 inference runs. Use high-precision timing (time.perf_counter) to capture start/end times for each run. Calculate comprehensive statistics: mean, median, standard deviation, min/max, percentiles (P95, P99), and throughput (FPS). This provides a complete performance profile showing typical inference time and worst-case scenarios.

**Performance Targets** (CPU-only, single thread):
| Device | Target Latency | Status |
|--------|----------------|--------|
| Desktop CPU | < 100 ms | ✅ Expected |
| High-end Mobile (Pixel 6, S21) | < 300 ms | 🎯 Target |
| Mid-range Mobile (Moto G, A52) | < 500 ms | ⚠️ Acceptable |
| Low-end Mobile | < 1000 ms | ❌ Too slow |

---

### 3.3 Model Size Verification

**Implementation**:
Create a utility function that iterates through all model files, checks their existence, calculates file sizes in both MB and KB, and displays a formatted comparison table. Compare across all formats: Original PyTorch (.pth), Optimized PyTorch (GAP), ONNX (.onnx), TensorFlow SavedModel, TFLite Float32, and TFLite INT8.

**Size Targets**:

- **Optimized PyTorch**: 1.5-2 MB
- **TFLite INT8**: < 1 MB (ideal: 500-800 KB)

---

## 🎯 Performance Targets

### Minimum Viable Product (MVP)

| Metric                | Target                | Critical?    |
| --------------------- | --------------------- | ------------ |
| **Model Size**        | < 2 MB                | ✅ Yes       |
| **Inference Latency** | < 300 ms (mobile CPU) | ✅ Yes       |
| **Accuracy Drop**     | < 5% from baseline    | ✅ Yes       |
| **Memory Usage**      | < 100 MB RAM          | ⚠️ Important |
| **Cold Start Time**   | < 500 ms              | ⚠️ Important |

### Optimistic Targets

| Metric                | Target        | Notes                       |
| --------------------- | ------------- | --------------------------- |
| **Model Size**        | < 1 MB        | With INT8 quantization      |
| **Inference Latency** | < 200 ms      | With NPU acceleration       |
| **Accuracy Drop**     | < 2%          | With knowledge distillation |
| **Battery Impact**    | < 5% per hour | Continuous inference        |

---

## ✅ Deployment Checklist

### Pre-Deployment (Model Side)

- [ ] **Phase 1.1**: Implement Global Average Pooling in model architecture
- [ ] **Phase 1.2**: Retrain model from scratch with GAP modification
- [ ] **Phase 1.3**: Validate accuracy on test set (target: ≥ original - 5%)
- [ ] **Phase 1.4**: Save optimized model weights (`best_convlstm_optimized.pth`)
- [ ] **Phase 2.1**: Export PyTorch model to ONNX format
- [ ] **Phase 2.2**: Verify ONNX model with `onnx.checker`
- [ ] **Phase 2.3**: Test ONNX inference with onnxruntime
- [ ] **Phase 2.4**: Convert ONNX to TensorFlow SavedModel
- [ ] **Phase 2.5**: Convert TensorFlow to TFLite (Float32)
- [ ] **Phase 2.6**: Apply INT8 quantization with representative dataset
- [ ] **Phase 3.1**: Run accuracy comparison across all formats
- [ ] **Phase 3.2**: Benchmark latency on CPU
- [ ] **Phase 3.3**: Verify model size < 2 MB (INT8 TFLite)
- [ ] **Phase 3.4**: Document accuracy metrics and performance benchmarks
- [ ] **Phase 3.5**: Create model documentation (input/output specs, preprocessing)

### Model Artifacts to Deliver

- [ ] `convlstm_model_int8.tflite` - Final mobile model
- [ ] `model_metadata.json` - Input/output specifications
- [ ] `preprocessing_specs.txt` - Frame extraction & normalization steps
- [ ] `performance_report.pdf` - Accuracy and latency benchmarks
- [ ] `test_samples/` - Sample inputs for Android integration testing

### Post-Deployment (Android Side - For Reference)

- [ ] Integrate TFLite Interpreter in Android app
- [ ] Implement video preprocessing (30 frames @ 10 FPS, 128×128 resize)
- [ ] Add intent signal generation logic
- [ ] Benchmark on-device latency
- [ ] Test with real-world driving videos
- [ ] Implement result visualization (turn predictions)

---

## 📚 Additional Resources

### Tools & Libraries

- **Netron**: Visualize model architectures ([https://netron.app/](https://netron.app/))
- **ONNX Runtime**: Test ONNX models ([https://onnxruntime.ai/](https://onnxruntime.ai/))
- **TensorFlow Lite**: Mobile ML framework ([https://www.tensorflow.org/lite](https://www.tensorflow.org/lite))
- **Android ML**: Official guide ([https://developer.android.com/ml](https://developer.android.com/ml))

### Best Practices

1. **Always retrain after architecture changes** (GAP, pruning)
2. **Use representative dataset from same distribution as production data**
3. **Test on actual mobile devices, not just emulators**
4. **Monitor battery consumption during continuous inference**
5. **Implement fallback mechanisms for low-accuracy predictions**

### Further Optimization

- **MobileNetV3**: Replace ConvLSTM backbone with efficient mobile architecture
- **Temporal Shift Module (TSM)**: Lightweight alternative to LSTM
- **Neural Architecture Search (NAS)**: Automatically find optimal mobile architecture
- **Edge TPU**: Deploy on Google Coral devices for ultra-low latency

---

## 📝 Summary

This guide provides a complete pipeline for deploying your ConvLSTM model to Android devices:

1. **Optimize Model**: Add Global Average Pooling (7.4 MB → 1.5 MB)
2. **Convert Formats**: PyTorch → ONNX → TensorFlow → TFLite
3. **Quantize**: Apply INT8 quantization (1.5 MB → <1 MB)
4. **Validate**: Ensure accuracy within 5% and latency < 300 ms
5. **Deploy**: Deliver `.tflite` model with preprocessing specifications

**Next Steps**: Start with Phase 1.1 (Global Average Pooling implementation) and proceed sequentially through the checklist.

---

**Last Updated**: February 20, 2026  
**Version**: 1.0  
**Contact**: For questions about mobile deployment, refer to this guide or consult TensorFlow Lite documentation.
