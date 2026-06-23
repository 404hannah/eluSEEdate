# Prototype 3 - TensorBoard Logging Feature (Preserved for Consideration)

## Note
This document preserves the TensorBoard logging feature discussion from an earlier prototype version. **I am currently unsure whether this feature is valuable enough to keep in the main training pipeline**, given the additional complexity and dependencies it introduces. Rather than lose the time spent coding and documenting this feature, it has been moved here for future reference while Prototype 5 continues development without it.

---

## TensorBoard Logging Feature

### What it is
TensorBoard is a visualization toolkit that provides an interactive web dashboard to visualize training metrics in real-time through charts and graphs. PyTorch supports it via `torch.utils.tensorboard.SummaryWriter`.

### How it's implemented
```python
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# TensorBoard Configuration
LOG_DIR = "runs"  # Directory for TensorBoard logs

# In main() function
run_name = f"convlstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))

# Log metrics after each epoch
writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss_avg}, epoch)
writer.add_scalars('Accuracy', {'Train': train_acc, 'Validation': val_acc}, epoch)
writer.add_scalar('Learning_Rate', current_lr, epoch)
writer.add_scalar('Gradient_Norm', avg_grad_norm, epoch)
writer.add_scalar('Inference/Latency_ms', avg_val_latency_ms, epoch)
writer.add_scalar('Inference/Throughput_batches_per_sec', val_throughput, epoch)

# Close writer at the end
writer.close()
```

### Why it helps
- **Problem solved:** Console-only logging provides no historical view of training progress. Once scrolled past, previous epoch data is lost, making it impossible to spot trends, compare runs, or create charts without manual data collection.
- **Impact:**
  - **Visual Loss Curves:** Instantly see if training is converging, diverging, or oscillating
  - **Learning Rate Tracking:** Visualize when ReduceLROnPlateau reduces LR
  - **Compare Runs:** Overlay different hyperparameter experiments
  - **Gradient Monitoring:** Spot gradient issues before they cause problems
  - **Real-time Updates:** Charts update live during training - no need to wait until completion
  - **Historical Record:** All runs are saved - review old experiments anytime
  - **Shareable:** Export charts for thesis documentation

### What we track
- `Loss/train` and `Loss/val` - Training and validation loss per epoch
- `Accuracy/train` and `Accuracy/val` - Training and validation accuracy per epoch
- `Learning_Rate` - Current LR (shows when scheduler reduces it)
- `Gradient_Norm` - Average gradient norm (shows if clipping is active)
- `Inference/Latency_ms` - Validation inference latency
- `Inference/Throughput_batches_per_sec` - Validation throughput

### How to view

**For Local Windows (VS Code):**
```powershell
# Run in VS Code PowerShell terminal (can be launched during or after training)
# Note that it will have empty data until first epoch finishes
py -m tensorboard.main --logdir=notebooks/runs

# Navigate to http://localhost:6006 in your browser
```

**For Google Colab (easier - displays inline):**
```python
# Load TensorBoard extension (run once at the start of your notebook)
%load_ext tensorboard

# Launch TensorBoard inline in the notebook (displays directly below the cell)
%tensorboard --logdir=runs
```

**Best practice:** Launch TensorBoard **while training is running** to see real-time chart updates as each epoch completes. You can also launch it after training finishes to review results.

### Prerequisites

**For Local Windows:**
```powershell
# Install TensorBoard (required for visualization)
py -m pip install tensorboard
```

**For Google Colab:**
```python
# TensorBoard is pre-installed on Colab, no installation needed!
# Just load the extension:
%load_ext tensorboard
```

### Configuration
```python
# TensorBoard Configuration
LOG_DIR = "runs"  # Directory for TensorBoard logs

# Launch TensorBoard viewer:
# Windows: py -m tensorboard.main --logdir=runs (then open http://localhost:6006)
# Colab: %tensorboard --logdir=runs (displays inline in notebook)
```

### Comparison to base model
The base model only printed metrics to console, providing no visualization, no historical comparison, and no way to share or export training progress.

---

## Considerations for Re-integration

**Pros:**
- Visual training analysis is valuable for debugging and understanding model behavior
- Historical record of all experiments
- Professional presentation for thesis documentation
- Industry-standard tool widely recognized in ML research

**Cons:**
- Adds external dependency (tensorboard package)
- Requires additional setup and configuration
- Not strictly necessary for model training to work
- Console logging may be sufficient for simpler workflows

**Decision pending:** Whether the visualization benefits outweigh the added complexity for this project's scope.

---

## References
- **TensorBoard:** Originally developed for TensorFlow by Google Brain team (2015), now supported natively by PyTorch via `torch.utils.tensorboard`

---

*Last Updated: February 17, 2026*
