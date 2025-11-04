#!/usr/bin/env python3
"""
Convert ONNX to TFLite using onnx-tf (Path B from solution.md)
"""
import os
import sys

print("="*70)
print("ONNX → TensorFlow → TFLite Conversion")
print("="*70)

# Check files
input_file = "face_parsing_sim.onnx"
if not os.path.exists(input_file):
    print(f"✗ {input_file} not found!")
    sys.exit(1)

size_mb = os.path.getsize(input_file) / (1024*1024)
print(f"\n✓ Input: {input_file} ({size_mb:.2f} MB)")

# Step 1: ONNX → TensorFlow SavedModel
print("\n[Step 1] Converting ONNX → TensorFlow SavedModel...")
try:
    import onnx
    from onnx_tf.backend import prepare
    
    onnx_model = onnx.load(input_file)
    print(f"  ONNX model loaded")
    
    tf_rep = prepare(onnx_model, auto_pad=True)
    print(f"  Prepared for TensorFlow")
    
    savedmodel_dir = "face_parsing_savedmodel"
    tf_rep.export_graph(savedmodel_dir)
    print(f"✓ SavedModel exported to: {savedmodel_dir}")
    
except Exception as e:
    print(f"✗ ONNX-TF conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: SavedModel → TFLite
print("\n[Step 2] Converting SavedModel → TFLite...")
try:
    import tensorflow as tf
    
    # Float32 version
    print("  Converting Float32 model...")
    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)
    
    # Add support for TensorFlow ops if needed
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    output_file = "face_parsing_float32.tflite"
    with open(output_file, "wb") as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"✓ TFLite model created: {output_file}")
    print(f"  Size: {size_kb:.2f} KB")
    
    # Try dynamic range quantization
    print("\n  Creating dynamic range quantized version...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        tflite_dynamic = converter.convert()
        
        dynamic_file = "face_parsing_dynamic.tflite"
        with open(dynamic_file, "wb") as f:
            f.write(tflite_dynamic)
        
        size_kb_dyn = len(tflite_dynamic) / 1024
        print(f"✓ Dynamic quantized model: {dynamic_file}")
        print(f"  Size: {size_kb_dyn:.2f} KB")
    except Exception as qe:
        print(f"⚠ Dynamic quantization skipped: {qe}")
    
    print("\n" + "="*70)
    print("CONVERSION SUCCESSFUL!")
    print("="*70)
    print(f"\n✓ Generated TFLite model: {output_file}")
    print(f"  Ready for mobile deployment!")
    
    # Verify the model
    print("\n[Verification] Testing TFLite model...")
    import numpy as np
    
    interpreter = tf.lite.Interpreter(model_path=output_file)
    interpreter.allocate_tensors()
    
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    
    print(f"  Input: {inp['name']} {inp['shape']} {inp['dtype']}")
    print(f"  Output: {out['name']} {out['shape']} {out['dtype']}")
    
    # Test with dummy data
    H, W = inp['shape'][1], inp['shape'][2]
    test_input = np.random.rand(1, H, W, 3).astype(np.float32)
    
    interpreter.set_tensor(inp['index'], test_input)
    interpreter.invoke()
    result = interpreter.get_tensor(out['index'])
    
    print(f"✓ Model runs successfully!")
    print(f"  Output shape: {result.shape}")
    
except Exception as e:
    print(f"✗ TFLite conversion failed: {e}")
    import traceback
    traceback.print_exc()
