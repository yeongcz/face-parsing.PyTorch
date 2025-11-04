#!/usr/bin/env python3
"""
Convert ONNX to TFLite using onnx2tf
"""
import os
import sys

print("="*70)
print("Step 2: Convert ONNX to TFLite")
print("="*70)

# First, check TensorFlow
print("\n[1] Checking TensorFlow installation...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
except Exception as e:
    print(f"✗ TensorFlow error: {e}")
    sys.exit(1)

# Import onnx2tf
print("\n[2] Loading onnx2tf...")
try:
    import onnx2tf
    print(f"✓ onnx2tf loaded")
except Exception as e:
    print(f"✗ onnx2tf error: {e}")
    print("\nTrying to fix by using convert function directly...")

# Convert
print("\n[3] Converting ONNX → TFLite...")
input_file = "face_parsing_sim.onnx"
output_dir = "face_parsing_tf"

if not os.path.exists(input_file):
    print(f"✗ {input_file} not found!")
    sys.exit(1)

try:
    # Try direct import and conversion
    from onnx2tf.onnx2tf import convert
    
    convert(
        input_onnx_file_path=input_file,
        output_folder_path=output_dir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=False
    )
    
    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {output_dir}")
    
    # List generated files
    if os.path.exists(output_dir):
        print("\nGenerated files:")
        for f in os.listdir(output_dir):
            fpath = os.path.join(output_dir, f)
            if os.path.isfile(fpath):
                size_kb = os.path.getsize(fpath) / 1024
                print(f"  {f} ({size_kb:.2f} KB)")
    
except Exception as e:
    print(f"\n✗ Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALTERNATIVE: Use onnx-tf instead")
    print("="*70)
    
    # Try onnx-tf as fallback
    try:
        print("\nAttempting onnx-tf conversion...")
        import onnx
        from onnx_tf.backend import prepare
        
        onnx_model = onnx.load(input_file)
        tf_rep = prepare(onnx_model, auto_pad=True)
        tf_rep.export_graph("face_parsing_savedmodel")
        print("✓ SavedModel exported")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model("face_parsing_savedmodel")
        tflite_model = converter.convert()
        
        tflite_path = "face_parsing_float32.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        
        size_kb = len(tflite_model) / 1024
        print(f"\n✓ TFLite model created: {tflite_path}")
        print(f"  Size: {size_kb:.2f} KB")
        
    except Exception as e2:
        print(f"✗ onnx-tf also failed: {e2}")
        print("\nPlease use Google Colab as recommended in solution.md")
