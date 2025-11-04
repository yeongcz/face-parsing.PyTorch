#!/usr/bin/env python3
"""
Convert ONNX model directly to TFLite format.

Step 3: ONNX → TFLite (Simple approach)
"""

import os
import sys
import tensorflow as tf
import onnx
import onnxruntime as rt
import numpy as np

def main():
    # Paths
    onnx_path = "face_parsing.onnx"
    tflite_path = "face_parsing.tflite"
    
    print("=" * 70)
    print("Step 3: ONNX → TFLite Conversion")
    print("=" * 70)
    
    # Step 1: Load and analyze ONNX model
    print("\n[Step 1] Loading and analyzing ONNX model...")
    print(f"Input: {onnx_path}")
    
    try:
        # Load ONNX model
        print(f"\nLoading ONNX model from: {onnx_path}")
        onnx_model = onnx.load(onnx_path)
        
        # Check model
        print("Checking ONNX model...")
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
        
        # Get model info
        graph = onnx_model.graph
        print(f"\nModel inputs:")
        for input_node in graph.input:
            shape = [d.dim_value for d in input_node.type.tensor_type.shape.dim]
            print(f"  - Name: {input_node.name}, Shape: {shape}")
        
        print(f"\nModel outputs:")
        for output_node in graph.output:
            shape = [d.dim_value for d in output_node.type.tensor_type.shape.dim]
            print(f"  - Name: {output_node.name}, Shape: {shape}")
        
    except Exception as e:
        print(f"\n✗ Failed to load ONNX model!")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Create a concrete function for TFLite
    print("\n" + "=" * 70)
    print("[Step 2] Converting ONNX to TFLite...")
    print("=" * 70)
    
    try:
        import onnxruntime as rt
        from tensorflow.python.framework.ops import EagerTensor
        
        # Create ONNX Runtime session
        print("\nCreating ONNX Runtime session...")
        sess = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Get input/output info
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        output_names = [output.name for output in sess.get_outputs()]
        
        print(f"Input: {input_name} with shape {input_shape}")
        print(f"Outputs: {output_names}")
        
        # Create a class to wrap the ONNX model
        class ONNXWrapper(tf.Module):
            def __init__(self, onnx_path):
                super(ONNXWrapper, self).__init__()
                self.sess = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                self.input_name = self.sess.get_inputs()[0].name
                self.output_names = [output.name for output in self.sess.get_outputs()]
            
            @tf.function(input_signature=[tf.TensorSpec([1, 3, 512, 512], tf.float32)])
            def __call__(self, input_tensor):
                # We need to process this in Python and return TensorFlow tensors
                py_outputs = tf.py_function(
                    func=self._run_inference,
                    inp=[input_tensor],
                    Tout=[tf.float32, tf.float32, tf.float32]
                )
                return py_outputs
            
            def _run_inference(self, input_tensor):
                input_data = input_tensor.numpy() if isinstance(input_tensor, EagerTensor) else np.array(input_tensor)
                outputs = self.sess.run(self.output_names, {self.input_name: input_data})
                return [np.array(output, dtype=np.float32) for output in outputs]
        
        print("\nCreating TensorFlow wrapper module...")
        wrapper = ONNXWrapper(onnx_path)
        
        # Create concrete function
        print("Creating concrete function...")
        concrete_func = wrapper.__call__.get_concrete_function(
            tf.TensorSpec([1, 3, 512, 512], tf.float32)
        )
        
        # Convert to TFLite
        print("\nConverting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.allow_custom_ops = True
        
        # This may fail if custom ops are needed, but let's try
        print("Generating TFLite model...")
        tflite_model = converter.convert()
        
        # Save TFLite model
        print(f"\nSaving TFLite model to: {tflite_path}")
        with open(tflite_path, "wb") as f:
            bytes_written = f.write(tflite_model)
            print(f"Wrote {bytes_written} bytes")
        
        print("✓ ONNX → TFLite conversion completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Conversion failed!")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Direct TFLite conversion of complex ONNX models can be challenging.")
        print("Alternative approaches:")
        print("  1. Convert ONNX → SavedModel → TFLite (requires compatible ops)")
        print("  2. Use MediaPipe model conversion tools")
        print("  3. Export model directly from PyTorch to TFLite format")
        return False
    
    # Final summary
    print("\n" + "=" * 70)
    print("Conversion Complete!")
    print("=" * 70)
    print(f"✓ Generated files:")
    print(f"  - {tflite_path}")
    
    # Check file size
    if os.path.exists(tflite_path):
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"\nTFLite model size: {size_mb:.2f} MB")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
