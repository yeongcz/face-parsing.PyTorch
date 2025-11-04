#!/usr/bin/env python3
"""
Direct PyTorch to TFLite using ai-edge-torch
Convert 79999_iter.pth to face_parsing.tflite
"""

import torch
import ai_edge_torch
import os

# Import model
from model import BiSeNet

print("="*70)
print("PyTorch → TFLite Conversion using ai-edge-torch")
print("="*70)

# Load model
print("\n[1] Loading PyTorch model...")
model_path = "79999_iter.pth"
n_classes = 19

net = BiSeNet(n_classes=n_classes)
net.load_state_dict(torch.load(model_path, map_location='cpu'))
net.eval()
print(f"✓ Model loaded: {model_path}")

# Create sample input
print("\n[2] Preparing sample input...")
sample_input = torch.randn(1, 3, 512, 512)
print(f"✓ Sample input shape: {sample_input.shape}")

# Convert to TFLite
print("\n[3] Converting to TFLite...")
try:
    edge_model = ai_edge_torch.convert(net, (sample_input,))
    
    # Save TFLite model
    tflite_path = "face_parsing.tflite"
    edge_model.export(tflite_path)
    
    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"✓ TFLite model saved: {tflite_path}")
    print(f"  Size: {size_kb:.2f} KB")
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)
    print(f"\n✓ Model: {tflite_path}")
    print(f"  Input: [1, 3, 512, 512] - RGB image")
    print(f"  Output: 19-class segmentation")
    print(f"\nReady for deployment on mobile devices!")
    
except Exception as e:
    print(f"✗ Conversion failed: {e}")
    print(f"\nError details: {type(e).__name__}")
    import traceback
    traceback.print_exc()
