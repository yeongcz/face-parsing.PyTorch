#!/usr/bin/env python3
"""
Export ONNX from PyTorch (following solution.md Path A)
"""
import torch
import onnx
from model import BiSeNet

print("="*70)
print("Step 1: Export ONNX from PyTorch")
print("="*70)

# Load checkpoint
print("\n[1] Loading PyTorch checkpoint...")
ckpt = torch.load("79999_iter.pth", map_location="cpu", weights_only=False)
net = BiSeNet(n_classes=19)
net.load_state_dict(ckpt)
net.eval()
print("✓ Model loaded")

# Export to ONNX with fixed input size
print("\n[2] Exporting to ONNX...")
dummy = torch.randn(1, 3, 512, 512)  # (N,C,H,W) fixed
torch.onnx.export(
    net, 
    dummy, 
    "face_parsing.onnx",
    input_names=["input"], 
    output_names=["logits"],
    opset_version=13,
    do_constant_folding=True,
    dynamic_axes=None  # keep static for TFLite
)

# Verify ONNX
print("\n[3] Verifying ONNX model...")
onnx_model = onnx.load("face_parsing.onnx")
onnx.checker.check_model(onnx_model)
print("✓ ONNX exported and checked")

import os
size_kb = os.path.getsize("face_parsing.onnx") / 1024
print(f"\n✓ Generated: face_parsing.onnx ({size_kb:.2f} KB)")
print("\nNext: Run simplification and conversion to TFLite")
