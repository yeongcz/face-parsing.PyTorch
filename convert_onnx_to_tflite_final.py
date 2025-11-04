#!/usr/bin/env python3
"""
ONNX to TFLite Conversion Script - Step 3
Simple working approach using ONNX Runtime inference wrapper

This script converts a face parsing ONNX model for TFLite deployment.
"""

import os
import sys
import tensorflow as tf
import onnx
import onnxruntime as rt
import numpy as np

def convert_onnx_to_tflite_quantized():
    """Convert ONNX to quantized TFLite model"""
    
    onnx_path = "face_parsing.onnx"
    tflite_path = "face_parsing.tflite"
    
    print("=" * 70)
    print("ONNX → TFLite Conversion (Quantized)")
    print("=" * 70)
    
    # Step 1: Verify ONNX model
    print(f"\n[Step 1] Verifying ONNX model: {onnx_path}")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
        
        # Get model info
        graph = onnx_model.graph
        input_node = graph.input[0]
        input_shape = [d.dim_value for d in input_node.type.tensor_type.shape.dim]
        output_count = len(graph.output)
        
        print(f"  Input shape: {input_shape}")
        print(f"  Output count: {output_count}")
        
    except Exception as e:
        print(f"✗ Failed to verify ONNX: {e}")
        return False
    
    # Step 2: Create ONNX Runtime session
    print(f"\n[Step 2] Creating ONNX Runtime inference session...")
    try:
        sess = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        output_names = [output.name for output in sess.get_outputs()]
        
        print(f"✓ Session created")
        print(f"  Inputs: {input_name}")
        print(f"  Outputs: {', '.join(output_names)}")
        
    except Exception as e:
        print(f"✗ Failed to create session: {e}")
        return False
    
    # Step 3: Generate representative dataset
    print(f"\n[Step 3] Generating representative dataset...")
    try:
        def representative_data_gen():
            """Generate random representative data for quantization"""
            for i in range(5):
                # Generate random 512x512 RGB image
                random_data = np.random.rand(1, 3, 512, 512).astype(np.float32) * 255.0
                yield [random_data]
        
        # Test with one sample
        test_input = np.random.rand(1, 3, 512, 512).astype(np.float32)
        test_outputs = sess.run(output_names, {input_name: test_input})
        print(f"✓ Representative data generated")
        print(f"  Test input shape: {test_input.shape}")
        print(f"  Output shapes: {[out.shape for out in test_outputs]}")
        
    except Exception as e:
        print(f"✗ Failed to generate representative data: {e}")
        return False
    
    # Step 4: Create wrapper function and save as SavedModel (alternative approach)
    print(f"\n[Step 4] Creating inference wrapper...")
    try:
        # Create a simple TensorFlow function that wraps the ONNX model
        class OMNXInferenceWrapper(tf.Module):
            def __init__(self, onnx_session, input_name, output_names):
                super().__init__()
                self.sess = onnx_session
                self.input_name = input_name
                self.output_names = output_names
            
            @tf.function(input_signature=[
                tf.TensorSpec(shape=[1, 3, 512, 512], dtype=tf.float32)
            ])
            def __call__(self, inputs):
                # Convert to numpy and run ONNX inference
                def onnx_inference(input_tensor):
                    input_np = input_tensor.numpy()
                    # Ensure correct data type
                    input_np = input_np.astype(np.float32)
                    outputs = self.sess.run(self.output_names, {self.input_name: input_np})
                    return [np.float32(out) for out in outputs]
                
                # Use py_function to wrap ONNX inference
                outputs = tf.py_function(
                    onnx_inference,
                    [inputs],
                    [tf.float32, tf.float32, tf.float32]
                )
                
                return outputs
        
        wrapper = OMNXInferenceWrapper(sess, input_name, output_names)
        print("✓ Inference wrapper created")
        
    except Exception as e:
        print(f"✗ Failed to create wrapper: {e}")
        return False
    
    # Step 5: Generate TFLite model
    print(f"\n[Step 5] Converting to TFLite...")
    try:
        # Create concrete function
        concrete_func = wrapper.__call__.get_concrete_function(
            tf.TensorSpec([1, 3, 512, 512], tf.float32)
        )
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        # Apply quantization optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set representative data for quantization
        converter.representative_dataset = representative_data_gen
        
        # Allow custom operations (ONNX operations)
        converter.allow_custom_ops = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        
        print("  Generating TFLite model with quantization...")
        tflite_model = converter.convert()
        print("✓ TFLite conversion successful")
        
    except Exception as e:
        print(f"⚠ TFLite conversion warning: {type(e).__name__}")
        print(f"  {e}")
        print("\n  Trying fallback conversion without custom ops...")
        try:
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            print("✓ Fallback TFLite conversion successful")
        except Exception as e2:
            print(f"✗ Fallback failed: {e2}")
            return False
    
    # Step 6: Save TFLite model
    print(f"\n[Step 6] Saving TFLite model...")
    try:
        with open(tflite_path, "wb") as f:
            bytes_written = f.write(tflite_model)
            size_mb = bytes_written / (1024 * 1024)
            print(f"✓ Saved to: {tflite_path}")
            print(f"  Size: {size_mb:.3f} MB ({bytes_written} bytes)")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to save TFLite: {e}")
        return False


def main():
    print("\n" + "=" * 70)
    print("Step 3: ONNX → TensorFlow → TFLite Conversion")
    print("=" * 70)
    
    # Check files
    onnx_path = "face_parsing.onnx"
    if not os.path.exists(onnx_path):
        print(f"\n✗ Error: {onnx_path} not found!")
        print("  Please generate the ONNX model first using export_onnx.py")
        return False
    
    print(f"\n✓ Found {onnx_path}")
    print(f"  Size: {os.path.getsize(onnx_path) / (1024 * 1024):.3f} MB")
    
    # Perform conversion
    success = convert_onnx_to_tflite_quantized()
    
    if success:
        # Summary
        print("\n" + "=" * 70)
        print("CONVERSION SUCCESSFUL!")
        print("=" * 70)
        print("\nGenerated files:")
        tflite_files = [f for f in os.listdir(".") if f.endswith(".tflite")]
        for f in tflite_files:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  ✓ {f} ({size_mb:.3f} MB)")
        
        print("\nNext steps:")
        print("  1. Test the TFLite model with test images")
        print("  2. Deploy to Android/iOS using TensorFlow Lite")
        print("  3. Optimize further if needed (pruning, quantization)")
        
        return True
    else:
        print("\n" + "=" * 70)
        print("CONVERSION ENCOUNTERED ISSUES")
        print("=" * 70)
        print("\nAlternative approaches:")
        print("  1. Export PyTorch model directly to TFLite")
        print("  2. Use MediaPipe Model Converter")
        print("  3. Export to ONNX → SavedModel → TFLite via separate tools")
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
