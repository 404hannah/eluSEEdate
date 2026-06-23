"""
Convert TensorFlow SavedModel to TensorFlow.js format
For mobile deployment in React Native/Expo app
"""

import os
import sys

def convert_model():
    """Convert TF SavedModel to TFJS Graph Model"""
    
    # Check if tensorflowjs is installed
    try:
        import tensorflowjs as tfjs
        print(f"✓ tensorflowjs version: {tfjs.__version__}")
    except ImportError:
        print("❌ tensorflowjs not installed")
        print("Installing tensorflowjs...")
        os.system("pip install tensorflowjs")
        import tensorflowjs as tfjs
    
    # Paths
    saved_model_dir = "tf_model"
    output_dir = "../test_deployment/assets/model"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📦 Converting model...")
    print(f"   Input: {saved_model_dir}")
    print(f"   Output: {output_dir}")
    
    try:
        # Convert SavedModel to TFJS Graph Model
        tfjs.converters.convert_tf_saved_model(
            saved_model_dir,
            output_dir,
            signature_name='serving_default',
            saved_model_tags='serve'
        )
        
        print("\n✅ Model converted successfully!")
        print(f"\n📁 Output files in {output_dir}:")
        
        # List output files
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   - {file} ({size_mb:.2f} MB)")
        
        print("\n✨ Model is ready to use in the mobile app!")
        print("   The app will automatically load it on next startup.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure TensorFlow SavedModel exists in conversion/tf_model/")
        print("2. Check that saved_model.pb file is present")
        print("3. Verify model was saved with correct signature")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TensorFlow.js Model Converter")
    print("=" * 60)
    
    # Change to conversion directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    success = convert_model()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ CONVERSION COMPLETE")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ CONVERSION FAILED")
        print("=" * 60)
        sys.exit(1)
