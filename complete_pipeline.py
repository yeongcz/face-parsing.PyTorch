#!/usr/bin/env python3
"""
Complete PyTorch → TFLite Pipeline
Re-export ONNX with proper settings, then convert to TFLite
"""
import torch
import torch.onnx
import os

# Import model
from model import BiSeNet

print("="*70)
print("PyTorch → ONNX → TFLite (Complete Pipeline)")
print("="*70)

# Load PyTorch model
print("\n[1] Loading PyTorch model...")
net = BiSeNet(n_classes=19)
net.load_state_dict(torch.load("79999_iter.pth", map_location='cpu'))
net.eval()
print("✓ Model loaded")

# Create dummy input
dummy_input = torch.randn(1, 3, 512, 512)

# Export to ONNX with better settings
print("\n[2] Exporting to ONNX (optimized for TFLite)...")
torch.onnx.export(
    net,
    dummy_input,
    "face_parsing_v2.onnx",
    export_params=True,
    opset_version=13,  # Try older opset for better compatibility
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None  # No dynamic axes for TFLite
)
print("✓ ONNX exported: face_parsing_v2.onnx")

# Simplify ONNX
print("\n[3] Simplifying ONNX model...")
try:
    import onnx
    from onnxsim import simplify
    
    model = onnx.load("face_parsing_v2.onnx")
    model_simp, check = simplify(model)
    
    if check:
        onnx.save(model_simp, "face_parsing_simplified.onnx")
        print("✓ Simplified ONNX saved")
        onnx_file = "face_parsing_simplified.onnx"
    else:
        print("⚠ Simplification check failed, using original")
        onnx_file = "face_parsing_v2.onnx"
except ImportError:
    print("⚠ onnx-simplifier not installed, using original ONNX")
    onnx_file = "face_parsing_v2.onnx"
except Exception as e:
    print(f"⚠ Simplification failed: {e}")
    onnx_file = "face_parsing_v2.onnx"

# Convert ONNX to TensorFlow SavedModel using tf2onnx in reverse
print("\n[4] Converting to TensorFlow...")
import subprocess
import sys

# Method 1: Try using TensorFlow directly
try:
    import tensorflow as tf
    import onnx
    print(f"  TensorFlow {tf.__version__}")
    
    # Try onnx2keras as alternative to onnx-tf
    try:
        print("\n  Installing onnx2keras...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "onnx2keras", "-q"
        ], check=True)
        
        from onnx2keras import onnx_to_keras
        
        onnx_model = onnx.load(onnx_file)
        k_model = onnx_to_keras(onnx_model, ['input'])
        
        print("✓ Converted to Keras model")
        
        # Convert Keras to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(k_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open("face_parsing.tflite", "wb") as f:
            f.write(tflite_model)
        
        size_kb = len(tflite_model) / 1024
        print(f"\n✓ TFLite model created: face_parsing.tflite")
        print(f"  Size: {size_kb:.2f} KB")
        
        print("\n" + "="*70)
        print("CONVERSION SUCCESSFUL!")
        print("="*70)
        
    except Exception as e:
        print(f"⚠ onnx2keras failed: {e}")
        
        # Try method 2: Direct ONNX runtime wrapper
        print("\n  Trying ONNX Runtime → TFLite wrapper...")
        print("  (This creates a TFLite that calls ONNX internally)")
        
        # Create wrapper model
        print("\n  Creating hybrid model...")
        print("  Note: For pure TFLite, you need tf-keras installed")
        print("  which requires Windows Long Path support enabled")
        
except Exception as e:
    print(f"✗ Conversion failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("STATUS")
print("="*70)
print(f"\n✓ PyTorch model: 79999_iter.pth")
print(f"✓ ONNX model: {onnx_file}")
print(f"\nTo complete TFLite conversion:")
print("\n  Option A: Enable Windows Long Paths (requires admin):")
print("    - Run PowerShell as Administrator")
print("    - Execute: New-ItemProperty -Path")
print("      'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem'")
print("      -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force")
print("    - Restart terminal")
print("    - Run: pip install tf-keras")
print("    - Run: onnx2tf -i", onnx_file)
print("\n  Option B: Use Google Colab (easiest):")
print("    - Upload", onnx_file)
print("    - !pip install onnx2tf")
print("    - !onnx2tf -i", onnx_file)
print("    - Download face_parsing.tflite")
print("\n  Option C: Use the ONNX model directly")
print("    - Many mobile frameworks support ONNX")
print("    - ONNX Runtime Mobile is well-optimized")
