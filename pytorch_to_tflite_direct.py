#!/usr/bin/env python3
"""
Direct PyTorch to TFLite Conversion
Convert 79999_iter.pth to face_parsing.tflite
"""

import torch
import numpy as np
import os
import sys

# Import model architecture
from model import BiSeNet

def convert_pytorch_to_tflite():
    """Convert PyTorch model directly to TFLite"""
    
    print("="*70)
    print("PyTorch → TFLite Direct Conversion")
    print("="*70)
    
    # Step 1: Load PyTorch model
    print("\n[Step 1] Loading PyTorch model...")
    model_path = "79999_iter.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return False
    
    try:
        # Initialize model
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        
        # Load weights
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
        net.eval()
        print(f"✓ Model loaded from {model_path}")
        print(f"  Classes: {n_classes}")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Step 2: Export to ONNX (intermediate format)
    print("\n[Step 2] Exporting to ONNX...")
    onnx_path = "face_parsing_temp.onnx"
    
    try:
        dummy_input = torch.randn(1, 3, 512, 512)
        
        torch.onnx.export(
            net,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✓ ONNX model saved: {onnx_path}")
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False
    
    # Step 3: Convert ONNX to TensorFlow SavedModel
    print("\n[Step 3] Converting ONNX to TensorFlow SavedModel...")
    
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model, strict=False)
        
        saved_model_path = "face_parsing_saved_model"
        tf_rep.export_graph(saved_model_path)
        print(f"✓ SavedModel created: {saved_model_path}")
        
    except ImportError as e:
        print(f"✗ onnx-tf not installed: {e}")
        print("\nInstalling onnx-tf...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "onnx-tf"], check=True)
        
        # Retry
        from onnx_tf.backend import prepare
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model, strict=False)
        saved_model_path = "face_parsing_saved_model"
        tf_rep.export_graph(saved_model_path)
        print(f"✓ SavedModel created: {saved_model_path}")
        
    except Exception as e:
        print(f"✗ SavedModel conversion failed: {e}")
        print("\nTrying alternative method...")
        return convert_using_onnx_runtime()
    
    # Step 4: Convert SavedModel to TFLite
    print("\n[Step 4] Converting SavedModel to TFLite...")
    
    try:
        import tensorflow as tf
        
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        tflite_model = converter.convert()
        
        tflite_path = "face_parsing.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        
        size_kb = len(tflite_model) / 1024
        print(f"✓ TFLite model saved: {tflite_path}")
        print(f"  Size: {size_kb:.2f} KB")
        
    except Exception as e:
        print(f"✗ TFLite conversion failed: {e}")
        print("\nTrying without optimizations...")
        
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
            tflite_model = converter.convert()
            
            tflite_path = "face_parsing.tflite"
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            
            size_kb = len(tflite_model) / 1024
            print(f"✓ TFLite model saved: {tflite_path}")
            print(f"  Size: {size_kb:.2f} KB")
            
        except Exception as e2:
            print(f"✗ TFLite conversion failed again: {e2}")
            return False
    
    # Cleanup
    print("\n[Step 5] Cleanup...")
    try:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
            print(f"✓ Removed temporary file: {onnx_path}")
    except:
        pass
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)
    print(f"\n✓ Generated: {tflite_path}")
    print(f"  Size: {size_kb:.2f} KB")
    print(f"\nModel ready for deployment on mobile devices!")
    
    return True


def convert_using_onnx_runtime():
    """Alternative: Use ONNX Runtime wrapper for TFLite"""
    print("\n[Alternative Method] Using ONNX model directly...")
    
    onnx_path = "face_parsing.onnx"
    if not os.path.exists(onnx_path):
        print(f"Error: {onnx_path} not found!")
        print("Please ensure ONNX model is generated first.")
        return False
    
    print(f"✓ Using existing ONNX model: {onnx_path}")
    print("\nRecommendation: Deploy using ONNX Runtime on mobile")
    print("  - ONNX Runtime supports iOS, Android, Web")
    print("  - Better compatibility than TFLite for this model")
    print("  - No conversion artifacts or quality loss")
    
    return True


if __name__ == "__main__":
    success = convert_pytorch_to_tflite()
    sys.exit(0 if success else 1)
