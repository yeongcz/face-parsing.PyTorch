#!/usr/bin/env python3
"""
Direct conversion using TensorFlow Lite converter
"""
import os
import numpy as np
import onnxruntime as ort

print("="*70)
print("Creating TFLite Model from ONNX")
print("="*70)

# Load ONNX model
print("\n[1] Loading ONNX model...")
onnx_model = "face_parsing.onnx"
session = ort.InferenceSession(onnx_model)

# Get input/output info
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_names = [out.name for out in session.get_outputs()]

print(f"✓ ONNX model loaded")
print(f"  Input: {input_name} {input_shape}")
print(f"  Outputs: {len(output_names)}")

# Create a simple TFLite model wrapper
print("\n[2] Creating TFLite conversion script...")

conversion_script = """
import tensorflow as tf
import onnxruntime as ort
import numpy as np

# Create a TF model that wraps ONNX inference
class ONNXWrapper(tf.Module):
    def __init__(self, onnx_path):
        super().__init__()
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 3, 512, 512], dtype=tf.float32)])
    def __call__(self, x):
        # Note: This won't actually work for TFLite as ONNX can't be embedded
        # We need a different approach
        return x

# This approach won't work - we need to use onnx-tensorflow properly
print("This approach requires loading ONNX into TensorFlow graph first...")
print("Attempting alternative conversion...")

# Alternative: Try using saved model approach
try:
    import onnx
    from onnx_tf.backend import prepare
    
    onnx_model = onnx.load("face_parsing.onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("saved_model")
    
    print("✓ Exported to SavedModel format")
    
    # Convert SavedModel to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open("face_parsing.tflite", "wb") as f:
        f.write(tflite_model)
    
    print("✓ TFLite model created: face_parsing.tflite")
    print(f"  Size: {len(tflite_model)/1024:.2f} KB")
    
except Exception as e:
    print(f"✗ Conversion failed: {e}")
    import traceback
    traceback.print_exc()
"""

with open("tf_convert.py", "w") as f:
    f.write(conversion_script)

print("✓ Created tf_convert.py")

# Actually, let's try a simpler working approach using openvino or direct methods
print("\n[3] Attempting direct conversion...")
print("\nNote: The standard ONNX→TFLite path requires tf-keras which has")
print("path length issues on Windows. Let me try using tf2onnx in reverse...")

print("\n" + "="*70)
print("Alternative: Using PyTorch model directly with ai-edge-torch")
print("="*70)

# Check if we can use tensorflow without tf-keras for basic conversion
try:
    import tensorflow as tf
    print(f"\n✓ TensorFlow {tf.__version__} available")
    
    # Try creating a simple converter from ONNX runtime results
    print("\nAttempting to trace ONNX execution pattern...")
    
    # This is a workaround - we'll create a TF function that mimics the model
    # but this won't give us the actual weights
    print("\n⚠ Standard conversion paths blocked by tf-keras dependency")
    print("\nRecommended solutions:")
    print("  1. Enable Windows Long Paths (run as Administrator):")
    print("     New-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force")
    print("\n  2. Use Google Colab or Linux system for conversion")
    print("\n  3. Use the ONNX model directly (many mobile frameworks support ONNX)")
    
except Exception as e:
    print(f"✗ Error: {e}")
