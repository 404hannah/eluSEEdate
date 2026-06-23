"""
Example: Processing Pre-Recorded Videos
========================================

This example demonstrates how to use the VideoPreprocessor to process
pre-recorded video files (MP4, AVI, MOV, etc.) for ConvLSTM turn prediction.

Use Case:
- Testing the preprocessing pipeline
- Offline video analysis
- Batch processing multiple videos
- Debugging and development

Date: February 25, 2026
"""

import numpy as np
import torch
import yaml
from pathlib import Path

# Import from mobile_app module
from preprocessor import VideoPreprocessor, preprocess_for_inference
from config import CONFIG

# Optional: Import model for full inference pipeline
# This will be imported dynamically in Example 5 when needed
# from models.conv_lstm_classifier import ConvLSTMModel

def example_basic_preprocessing():
    """
    Example 1: Basic video preprocessing
    
    Demonstrates the simplest way to preprocess a video file
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Video Preprocessing")
    print("=" * 70)
    
    # Initialize preprocessor with default settings
    preprocessor = VideoPreprocessor()
    
    # Path to your video file
    video_path = "path/to/your/video.mp4"  # Replace with actual path
    
    # Preprocess video with left turn intent at frame 5
    video_tensor = preprocessor.preprocess_video(
        video_path=video_path,
        intent=1,  # 0=Front, 1=Left, 2=Right
        intent_position=5  # Frame where direction was detected
    )
    
    # Check output shape
    print(f"✓ Input video: {video_path}")
    print(f"✓ Output shape: {video_tensor.shape}")  # Should be (20, 6, 128, 128)
    print(f"✓ Data type: {video_tensor.dtype}")  # float32
    print(f"✓ Value range: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]")
    print(f"✓ Intent channel active at frames 5-14: Left (channel 4)")
    
    # Verify intent encoding
    intent_channel = 4  # Left turn is channel 4 (index 1 + 3)
    for i in range(5, 15):  # Frames 5-14 should have direction detection
        assert video_tensor[i, intent_channel, :, :].max() == 1.0
    print("✓ Intent encoding validated.")
    
    print()


def example_convenience_function():
    """
    Example 2: Using the convenience function
    
    The preprocess_for_inference() function adds batch dimension automatically
    """
    print("=" * 70)
    print("EXAMPLE 2: Using Convenience Function")
    print("=" * 70)
    
    video_path = "path/to/your/video.mp4"  # Replace with actual path
    
    # One-line preprocessing with batch dimension
    video_tensor = preprocess_for_inference(
        video_path=video_path,
        intent=2,  # Right turn
        intent_position=10  # Intent starts at frame 10
    )
    
    print(f"✓ Output shape: {video_tensor.shape}")  # Should be (1, 20, 6, 128, 128)
    print(f"✓ Batch dimension added automatically")
    print(f"✓ Ready for direct model inference.")
    
    print()

def example_no_intent():
    """
    Example 3: Processing video without intent detection
    
    Useful for testing videos where no direction intent is present
    """
    print("=" * 70)
    print("EXAMPLE 3: No Intent Detection")
    print("=" * 70)
    
    preprocessor = VideoPreprocessor()
    video_path = "path/to/your/video.mp4"  # Replace with actual path
    
    # Process without intent
    video_tensor = preprocessor.preprocess_video(
        video_path=video_path,
        intent=None,  # No intent
        intent_position=None
    )
    
    print(f"✓ Output shape: {video_tensor.shape}")  # Still (20, 6, 128, 128)
    print(f"✓ Intent channels (3-5) should be all zeros")
    
    # Verify no intent signal
    intent_channels = video_tensor[:, 3:6, :, :]  # Extract intent channels
    assert intent_channels.max() == 0.0
    print("✓ No intent signal present (all zeros)")
    
    print()


def example_batch_processing():
    """
    Example 4: Batch processing multiple videos
    
    Process multiple videos efficiently in a loop
    """
    print("=" * 70)
    print("EXAMPLE 4: Batch Processing Multiple Videos")
    print("=" * 70)
    
    # List of videos to process
    video_files = [
        ("video1.mp4", 0, 5),   # (path, intent, position)
        ("video2.mp4", 1, 3),
        ("video3.mp4", 2, 8),
        ("video4.mp4", None, None),  # No intent
    ]
    
    preprocessor = VideoPreprocessor()
    processed_videos = []
    
    for video_path, intent, intent_position in video_files:
        try:
            video_tensor = preprocessor.preprocess_video(
                video_path=video_path,
                intent=intent,
                intent_position=intent_position
            )
            processed_videos.append(video_tensor)
            
            intent_str = CONFIG.class_names[intent] if intent is not None else "None"
            print(f"✓ Processed: {video_path} | Intent: {intent_str} | Shape: {video_tensor.shape}")
            
        except Exception as e:
            print(f"✗ Failed to process {video_path}: {e}")
    
    # Stack into batch
    if processed_videos:
        batch_tensor = np.stack(processed_videos, axis=0)
        print(f"\n✓ Batch shape: {batch_tensor.shape}")  # (num_videos, 20, 6, 128, 128)
    
    print()


def example_full_inference_pipeline():
    """
    Example 5: Complete inference pipeline
    
    Full workflow from video file to prediction result
    """
    print("=" * 70)
    print("EXAMPLE 5: Full Inference Pipeline")
    print("=" * 70)
    
    # Step 1: Preprocess video
    video_path = "path/to/your/video.mp4"  # Replace with actual path
    
    try:
        video_tensor = preprocess_for_inference(
            video_path=video_path,
            intent=1,  # Left turn
            intent_position=7
        )
        print(f"✓ Step 1: Video preprocessed | Shape: {video_tensor.shape}")
    except Exception as e:
        print(f"✗ Step 1 failed: {e}")
        print("  Make sure to replace 'path/to/your/video.mp4' with a real video file")
        print()
        return
    
    # Step 2: Convert to PyTorch tensor
    video_tensor = torch.from_numpy(video_tensor).float()
    print(f"✓ Step 2: Converted to PyTorch tensor | Device: {video_tensor.device}")
    
    # Step 3: Load model configuration and architecture
    try:
        # Add parent directory to path to import model
        import sys
        sys.path.append('../')
        from models.conv_lstm_classifier import ConvLSTMModel
        
        # Load configuration
        with open('model_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize model
        model = ConvLSTMModel(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            kernel_size=tuple(config['model']['kernel_size']),
            num_layers=config['model']['num_layers'],
            height=config['model']['height'],
            width=config['model']['width'],
            num_classes=config['model']['num_classes']
        )
        
        print("✓ Step 3: Model architecture initialized")
        
    except FileNotFoundError as e:
        print(f"✗ Step 3 failed: Configuration file not found")
        print(f"  Make sure 'model_config.yaml' exists in the mobile_app directory")
        print()
        return
    except ImportError as e:
        print(f"✗ Step 3 failed: Could not import model architecture")
        print(f"  Make sure '../models/conv_lstm_classifier.py' exists")
        print()
        return
    except Exception as e:
        print(f"✗ Step 3 failed: {e}")
        print()
        return
    
    # Step 4: Load trained weights
    try:
        model.load_state_dict(torch.load("../notebooks/best_convlstm.pth", map_location='cpu'))
        model.eval()
        print("✓ Step 4: Model weights loaded and set to evaluation mode")
        
    except FileNotFoundError:
        print("✗ Step 4 failed: Model weights file not found")
        print("  Make sure '../notebooks/best_convlstm.pth' exists")
        print("  You need to train the model first using train.py")
        print()
        return
    except Exception as e:
        print(f"✗ Step 4 failed: {e}")
        print()
        return
    
    # Step 5: Make prediction
    try:
        with torch.no_grad():
            prediction = model(video_tensor)
            predicted_class = prediction.argmax(dim=1).item()
            predicted_label = CONFIG.class_names[predicted_class]
            confidence = torch.softmax(prediction, dim=1)[0, predicted_class].item()
        
        print(f"✓ Step 5: Prediction made")
        print(f"   Predicted direction: {predicted_label}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Raw logits: {prediction[0].tolist()}")
        
    except Exception as e:
        print(f"✗ Step 5 failed: {e}")
        print()
        return
    
    print("\n✓ Full inference pipeline completed successfully.")
    print()

def example_custom_configuration():
    """
    Example 6: Custom preprocessing configuration
    
    Override default settings for specific use cases
    """
    print("=" * 70)
    print("EXAMPLE 6: Custom Configuration")
    print("=" * 70)
    
    # Create preprocessor with custom settings
    custom_preprocessor = VideoPreprocessor(
        height=64,         # Lower resolution for faster processing
        width=64,
        fps=5,             # Lower frame rate (5 FPS instead of 10)
        seq_len=10,        # Shorter sequences (1 second instead of 2)
        normalize=True
    )
    
    print(f"✓ Custom configuration:")
    print(f"  Resolution: {custom_preprocessor.width}x{custom_preprocessor.height}")
    print(f"  FPS: {custom_preprocessor.fps}")
    print(f"  Sequence length: {custom_preprocessor.seq_len} frames")
    print(f"  Expected output shape: {custom_preprocessor.get_output_shape()}")
    
    # Note: This would require retraining the model with matching dimensions
    print("\n⚠ WARNING: Custom dimensions require model retraining.")
    print("  Default (128x128, 10 FPS, 20 frames) matches Prototype 9 model")
    
    print()

def example_validation():
    """
    Example 7: Validate preprocessing output
    
    Checks to ensure preprocessing output is correct
    """
    print("=" * 70)
    print("EXAMPLE 7: Output Validation")
    print("=" * 70)
    
    video_path = "path/to/your/video.mp4"  # Replace with actual path
    preprocessor = VideoPreprocessor()
    
    # Process video
    video_tensor = preprocessor.preprocess_video(
        video_path=video_path,
        intent=1,
        intent_position=5
    )
    
    # Validation checks
    checks = []
    
    # Check 1: Shape
    expected_shape = (20, 6, 128, 128)
    checks.append(("Shape", video_tensor.shape == expected_shape, f"{video_tensor.shape}"))
    
    # Check 2: Data type
    checks.append(("Data type", video_tensor.dtype == np.float32, f"{video_tensor.dtype}"))
    
    # Check 3: Value range (normalized)
    in_range = (video_tensor.min() >= 0.0) and (video_tensor.max() <= 1.0)
    checks.append(("Value range [0,1]", in_range, f"[{video_tensor.min():.3f}, {video_tensor.max():.3f}]"))
    
    # Check 4: RGB channels (should have non-zero values)
    rgb_has_data = video_tensor[:, 0:3, :, :].max() > 0.0
    checks.append(("RGB channels populated", rgb_has_data, "✓"))
    
    # Check 5: Intent channels (one should be active for frames 5-14)
    intent_active = video_tensor[5:15, 4, :, :].max() == 1.0  # Left turn (channel 4)
    checks.append(("Intent detection present", intent_active, "Frames 5-14, Channel 4"))
    
    # Check 6: Intent outside range should be zero
    intent_inactive = video_tensor[0:5, 3:6, :, :].max() == 0.0
    checks.append(("Intent inactive before position", intent_inactive, "Frames 0-4"))
    
    # Print results
    print("Validation Results:")
    print("-" * 70)
    for check_name, passed, info in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | {check_name:30s} | {info}")
    
    all_passed = all(check[1] for check in checks)
    print("-" * 70)
    if all_passed:
        print("✓ All validation checks passed.")
    else:
        print("✗ Some validation checks failed - review preprocessing")
    
    print()

def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("VIDEO PREPROCESSING EXAMPLES - PRE-RECORDED VIDEOS")
    print("ConvLSTM Turn Prediction - Mobile App Module")
    print("=" * 70)
    print()
    
    # Display configuration
    print("Current Configuration:")
    print(f"  Resolution: {CONFIG.width}x{CONFIG.height}")
    print(f"  FPS: {CONFIG.fps}")
    print(f"  Sequence Length: {CONFIG.seq_len} frames ({CONFIG.duration} seconds)")
    print(f"  Channels: {CONFIG.channels} (3 RGB + 3 Intent)")
    print(f"  Classes: {', '.join(CONFIG.class_names)}")
    print()
    
    # Note about example videos
    print("⚠ NOTE: Examples use placeholder video paths")
    print("  Replace 'path/to/your/video.mp4' with actual video files")
    print()
    
    # Run examples
    try:
        example_basic_preprocessing()
    except Exception as e:
        print(f"⚠ Example 1 skipped: {e}\n")
    
    try:
        example_convenience_function()
    except Exception as e:
        print(f"⚠ Example 2 skipped: {e}\n")
    
    try:
        example_no_intent()
    except Exception as e:
        print(f"⚠ Example 3 skipped: {e}\n")
    
    try:
        example_batch_processing()
    except Exception as e:
        print(f"⚠ Example 4 skipped: {e}\n")
    
    example_full_inference_pipeline()
    
    example_custom_configuration()
    
    try:
        example_validation()
    except Exception as e:
        print(f"⚠ Example 7 skipped: {e}\n")
    
    print("=" * 70)
    print("Examples Complete.")
    print("=" * 70)
    print()

if __name__ == "__main__":
    main()