import torch as torch
import torch.nn as nn
from typing import List, Tuple, Optional, Any

# ConvLSTM Architecture for Video Direction Classification
# PROTOTYPE 9 - Mobile-Optimized with Global Average Pooling
#
# MODEL REQUIREMENTS:
# -------------------
# Input Shape:                      [Batch, Seq_Len, Channels, Height, Width]
#                                   [B, 20, 6, 128, 128] - 20 frames, 6 channels (RGB + Intent)
#
# LAYER STRUCTURE (Config: num_layers=2, hidden_dim=[64, 32]):
# -------------------
# Layer 1: ConvLSTMCell
#   - Input channels:               6 (RGB + 3 intent channels)
#   - Hidden channels:              64
#   - Kernel size:                  3×3
#   - Output:                       [B, 20, 64, 128, 128]
#
# Layer 2: ConvLSTMCell
#   - Input channels:               64 (from Layer 1)
#   - Hidden channels:              32
#   - Kernel size:                  3×3
#   - Output:                       [B, 20, 32, 128, 128]
#
# Classification Head (Mobile-Optimized):
#   - Takes last time step:         [B, 32, 128, 128]
#   - Global Average Pool:          [B, 32, 1, 1]
#   - Flatten:                      [B, 32]
#   - Dropout (0.5)
#   - Linear layer:                 [32 → 3]
#   - Output logits:                [B, 3] (Front, Left, Right)
#
# KERNEL OPERATIONS:
# -------------------
# Each ConvLSTMCell uses 3×3 convolutions with:
#   - Padding: 1 (preserves spatial dimensions)
#   - 4 parallel convolutions for LSTM gates: [input, forget, output, cell]
#   - Total parameters per cell: (input_dim + hidden_dim) × 4 × hidden_dim × 3 × 3
#
# Example Layer 1: (6 + 64) × 4 × 64 × 9 = 161,280 parameters
# Example Layer 2: (64 + 32) × 4 × 32 × 9 = 110,592 parameters

class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM memory unit - processes one frame at a time.

    Combines convolutional feature extraction with LSTM temporal memory.
    Uses 4 gates (input, forget, output, cell) to control information flow.

    Args:
        input_dim: Number of input channels (e.g., 3 for RGB, 64 for hidden layer)
        hidden_dim: Number of hidden state channels (e.g., 64, 32)
        kernel_size: Size of convolutional kernel (e.g., (3, 3))
        bias: Whether to use bias in convolutions
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int],
        bias: bool = True
    ) -> None:
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        cur_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one frame through ConvLSTM cell.

        LSTM Gate Equations:
        --------------------
        i_t = σ(W_xi * x_t + W_hi * h_{t-1})  ← Input gate (what to add)
        f_t = σ(W_xf * x_t + W_hf * h_{t-1})  ← Forget gate (what to keep)
        o_t = σ(W_xo * x_t + W_ho * h_{t-1})  ← Output gate (what to expose)
        g_t = tanh(W_xg * x_t + W_hg * h_{t-1}) ← Cell candidate

        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t        ← Update cell state
        h_t = o_t ⊙ tanh(c_t)                   ← Update hidden state

        where ⊙ = element-wise multiplication, σ = sigmoid
        """
        h_cur, c_cur = cur_state  # Extract previous hidden and cell states

        # Concatenate input frame with previous hidden state
        # [B, input_dim, H, W] + [B, hidden_dim, H, W] → [B, input_dim+hidden_dim, H, W]
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # Apply convolution to produce 4 gate activations
        # [B, input_dim+hidden_dim, H, W] → [B, 4×hidden_dim, H, W]
        combined_conv = self.conv(combined)

        # Split into 4 separate gates, each with hidden_dim channels
        # [B, 4×hidden_dim, H, W] → 4 tensors of [B, hidden_dim, H, W]
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Apply activation functions to each gate
        i = torch.sigmoid(cc_i)  # Input gate: how much new info to add (0-1)
        f = torch.sigmoid(cc_f)  # Forget gate: how much old info to keep (0-1)
        o = torch.sigmoid(cc_o)  # Output gate: how much to expose (0-1)
        g = torch.tanh(cc_g)     # Cell candidate: new information (-1 to 1)

        # Update cell state: forget old + add new
        c_next = f * c_cur + i * g

        # Update hidden state: apply output gate to cell state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states with zeros.

        Returns:
            h_0: [batch_size, hidden_dim, height, width] - initial hidden state
            c_0: [batch_size, hidden_dim, height, width] - initial cell state

        Example for Layer 1: [B, 64, 128, 128]
        Example for Layer 2: [B, 32, 128, 128]
        """
        height, width = image_size
        device = self.conv.weight.device  # Match device of model parameters
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM - processes video sequences frame by frame.

    Stacks multiple ConvLSTMCell layers to form a deep temporal network.
    Each layer processes all frames sequentially, then passes to next layer.

    FORWARD PASS FLOW (num_layers=2, seq_len=30):
    ------------------------------------------------
    Input: [B, 30, 3, 128, 128]
      ↓
    Layer 1 processes frame-by-frame:
      ├─ Frame 1:  [B, 3, 128, 128] → h1, c1 [B, 64, 128, 128]
      ├─ Frame 2:  [B, 3, 128, 128] → h2, c2 [B, 64, 128, 128]
      ⋮
      └─ Frame 30: [B, 3, 128, 128] → h30, c30 [B, 64, 128, 128]
    Stack outputs → [B, 30, 64, 128, 128]
      ↓
    Layer 2 processes frame-by-frame:
      ├─ Frame 1:  [B, 64, 128, 128] → h1, c1 [B, 32, 128, 128]
      ├─ Frame 2:  [B, 64, 128, 128] → h2, c2 [B, 32, 128, 128]
      ⋮
      └─ Frame 30: [B, 64, 128, 128] → h30, c30 [B, 32, 128, 128]
    Stack outputs → [B, 30, 32, 128, 128]
      ↓
    Final output: Last layer's all time steps [B, 30, 32, 128, 128]

    Args:
        input_dim: Input channels (3 for RGB)
        hidden_dim: List of hidden dims per layer [64, 32]
        kernel_size: Kernel size for conv operations (3, 3)
        num_layers: Number of stacked ConvLSTM layers (2)
        batch_first: If True, input shape is [B, T, C, H, W]
        bias: Use bias in convolutions
        return_all_layers: If True, return outputs from all layers
    """

    def __init__(
        self,
        input_dim: int,               # 3 (RGB channels)
        hidden_dim: List[int],        # [64, 32] - one per layer
        kernel_size: Tuple[int, int], # (3, 3)
        num_layers: int,              # 2
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False
    ) -> None:
        super(ConvLSTM, self).__init__()

        # Validate and extend parameters for all layers
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)  # [(3,3), (3,3)]
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)    # [64, 32]

        self.input_dim = input_dim          # 3
        self.hidden_dim = hidden_dim        # [64, 32]
        self.kernel_size = kernel_size      # [(3,3), (3,3)]
        self.num_layers = num_layers        # 2
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Build list of ConvLSTM cells (one per layer)
        # Layer 0: input_dim=3, hidden_dim=64
        # Layer 1: input_dim=64, hidden_dim=32
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            ))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self,
        input_tensor: torch.Tensor,  # [B, 30, 3, 128, 128]
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through all ConvLSTM layers.

        Process:
        --------
        1. For each layer (2 layers):
           2. Initialize hidden states if not provided
           3. Process all 30 frames sequentially
           4. Stack frame outputs → [B, 30, hidden_dim, H, W]
           5. Use this as input to next layer

        Returns:
        --------
        layer_output_list: List of outputs from each layer
                          [[B, 30, 32, 128, 128]] if return_all_layers=False
                          [[B, 30, 64, 128, 128], [B, 30, 32, 128, 128]] if True

        last_state_list: List of final (h, c) for each layer
                        [(h_layer1, c_layer1), (h_layer2, c_layer2)]
        """
        # Convert to batch_first format if needed
        if not self.batch_first:
            # [T, B, C, H, W] → [B, T, C, H, W]
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Extract dimensions
        b, seq_len, _, h, w = input_tensor.size()  # [B=6, T=30, C=3, H=128, W=128]

        # Initialize hidden states for all layers if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor  # Start with raw video [B, 30, 3, 128, 128]

        # Process each layer sequentially
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]  # Get initial states for this layer
            output_inner = []  # Store outputs for all time steps

            # Process each frame (time step) sequentially
            # This is where the RECURRENCE happens
            for t in range(seq_len):
                # Extract frame t: [B, C, H, W]
                input_t = cur_layer_input[:, t, :, :, :]

                # Process through ConvLSTM cell
                # Input: [B, C_in, H, W] + previous (h, c)
                # Output: new (h, c) with shape [B, hidden_dim, H, W]
                h, c = self.cell_list[layer_idx](input_tensor=input_t, cur_state=[h, c])

                # Save hidden state for this time step
                output_inner.append(h)

            # Stack all time steps: list of [B, hidden_dim, H, W] → [B, T, hidden_dim, H, W]
            layer_output = torch.stack(output_inner, dim=1)

            # This becomes input to next layer
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # If only returning final layer output (default)
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]  # Keep only last layer
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(
        self,
        batch_size: int,
        image_size: Tuple[int, int]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden states for all layers."""
        return [self.cell_list[i].init_hidden(batch_size, image_size)
                for i in range(self.num_layers)]

    @staticmethod
    def _check_kernel_size_consistency(kernel_size: Any) -> None:
        """Validate kernel_size is tuple or list of tuples."""
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all(isinstance(elem, tuple) for elem in kernel_size))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param: Any, num_layers: int) -> List:
        """
        Extend parameter for all layers if single value provided.

        Example:
            (3, 3) with num_layers=2 → [(3, 3), (3, 3)]
            [64, 32] with num_layers=2 → [64, 32] (unchanged)
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTMModel(nn.Module):
    """
    ConvLSTM classifier - processes video and outputs class prediction.
    PROTOTYPE 9 - Mobile-Optimized with Global Average Pooling

    FULL PIPELINE:
    --------------
    Input Video: [B, 20, 6, 128, 128]
        ↓
    ConvLSTM (2 layers):
        Layer 1: [B, 20, 6, 128, 128] → [B, 20, 64, 128, 128]
        Layer 2: [B, 20, 64, 128, 128] → [B, 20, 32, 128, 128]
        ↓
    Take Last Time Step: [B, 20, 32, 128, 128] → [B, 32, 128, 128]
        ↓
    Global Average Pooling: [B, 32, 128, 128] → [B, 32, 1, 1]
        ↓
    Flatten: [B, 32, 1, 1] → [B, 32]
        ↓
    Dropout(0.5): [B, 32] → [B, 32]
        ↓
    Linear: [B, 32] → [B, 3]
        ↓
    Output Logits: [B, 3] (Front=0, Left=1, Right=2)

    PARAMETERS (Optimized for Mobile):
    -----------
    ConvLSTM layers: ~272K parameters
    Linear layer:    32 × 3 = 96 parameters
    Total:           ~272K parameters (-85% from original)
    Model size:      ~1.5 MB (vs 7.4 MB without GAP)

    Args:
        input_dim: Input channels (6 for RGB + Intent)
        hidden_dim: Hidden dims per layer [64, 32]
        kernel_size: Conv kernel size (3, 3)
        num_layers: Number of ConvLSTM layers (2)
        height: Frame height (128, kept for compatibility)
        width: Frame width (128, kept for compatibility)
        num_classes: Output classes (3)
        dropout_rate: Dropout probability (0.5)
    """

    def __init__(
        self,
        input_dim: int,               # 6 (RGB + Intent)
        hidden_dim: List[int],        # [64, 32]
        kernel_size: Tuple[int, int], # (3, 3)
        num_layers: int,              # 2
        height: int,                  # 128 (kept for compatibility, not used with GAP)
        width: int,                   # 128 (kept for compatibility, not used with GAP)
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False,
        num_classes: int = 3,        # Front, Left, Right
        dropout_rate: float = 0.5
    ) -> None:
        super(ConvLSTMModel, self).__init__()

        # Build the ConvLSTM backbone
        self.convlstm = ConvLSTM(
            input_dim, hidden_dim, kernel_size, num_layers,
            batch_first=batch_first, bias=bias,
            return_all_layers=return_all_layers
        )

        # Global Average Pooling (Mobile Optimization)
        # Reduces spatial dimensions [B, C, H, W] → [B, C, 1, 1]
        # Eliminates 99.99% of FC layer parameters (1,572,864 → 96)
        # Model size reduction: ~7.4 MB → ~1.5 MB (-80%)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Classification head: maps pooled features to class logits
        # Input size: hidden_dim[-1] = 32 (after GAP reduces [32, 128, 128] to [32, 1, 1])
        # Output size: num_classes = 3
        # Parameters: 32 × 3 + 3 = 99 (vs 1,572,867 in original model)
        self.linear = nn.Linear(hidden_dim[-1], num_classes)

    def forward(
        self,
        input_tensor: torch.Tensor,  # [B, 20, 6, 128, 128]
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Full forward pass: video → ConvLSTM → classification.

        Steps:
        ------
        1. Process video through ConvLSTM layers
           [B, 20, 6, 128, 128] → [B, 20, 32, 128, 128]

        2. Take last time step (frame 20 contains all temporal info)
           [B, 20, 32, 128, 128] → [B, 32, 128, 128]

        3. Apply Global Average Pooling (Mobile Optimization)
           [B, 32, 128, 128] → [B, 32, 1, 1]
           Aggregates spatial information into channel-wise features

        4. Flatten pooled features
           [B, 32, 1, 1] → [B, 32]

        5. Apply dropout (training only)
           [B, 32] → [B, 32]

        6. Linear classification layer
           [B, 32] → [B, 3]

        Returns:
        --------
        logits: [B, 3] - raw scores for each class (before softmax)
        """
        # Process through ConvLSTM layers
        x, _ = self.convlstm(input_tensor)  # x is list, x[0] = [B, 20, 32, 128, 128]

        # Extract last time step (final frame has seen all previous frames)
        last_time_step = x[0][:, -1, :, :, :]  # [B, 32, 128, 128]

        # Apply Global Average Pooling to reduce spatial dimensions
        # [B, 32, 128, 128] → [B, 32, 1, 1]
        # This aggregates spatial information into channel-wise features
        pooled = self.global_avg_pool(last_time_step)

        # Flatten pooled features for classification
        # [B, 32, 1, 1] → [B, 32]
        flattened = torch.flatten(pooled, start_dim=1)

        # Apply dropout (active during training, disabled during eval)
        flattened = self.dropout(flattened)

        # Final linear layer: produce class logits
        return self.linear(flattened)  # [B, 3]