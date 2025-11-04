# QUICK REFERENCE: ONNX → TFLite Conversion

## Current Status
```
✓ Step 1: PyTorch → ONNX          COMPLETE
✓ Step 2: ONNX Validation         COMPLETE
⚠ Step 3: ONNX → TensorFlow → TFLite    IN PROGRESS
```

## Generated Artifacts
```
face_parsing.onnx              111.73 KB    ← Main model, ready to convert
face_parsing.tflite            0.87 KB      ← Wrapper (needs full conversion)
STEP3_CONVERSION_SUMMARY.py    8.38 KB      ← Run for detailed guide
STEP3_ONNX_TO_TFLITE.md        6.30 KB      ← Conversion troubleshooting
README_CONVERSION.md           11.49 KB     ← Complete project docs
```

## Quick Start (Pick ONE)

### Option A: ONNX → SavedModel → TFLite (BEST)
```bash
# 1. Install
pip install onnx-tf tensorflow-probability[tf]

# 2. Convert to SavedModel
python3 << 'EOF'
import onnx
from onnx_tf.backend import prepare
onnx_model = onnx.load("face_parsing.onnx")
tf_rep = prepare(onnx_model, strict=False)
tf_rep.export_graph("face_parsing_saved_model")
EOF

# 3. Convert to TFLite
python3 << 'EOF'
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model("face_parsing_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("face_parsing.tflite", "wb").write(tflite_model)
EOF

# Result: face_parsing.tflite ready for mobile!
```

### Option B: Use ONNX Runtime (NO CONVERSION)
```bash
# 1. Install
pip install onnxruntime

# 2. Use directly
python3 << 'EOF'
import onnxruntime as rt
import numpy as np
sess = rt.InferenceSession("face_parsing.onnx")
test = np.random.rand(1, 3, 512, 512).astype(np.float32)
outputs = sess.run(None, {"input": test})
print([o.shape for o in outputs])
EOF
```

### Option C: Automated Script
```bash
# Run summary script for all methods
python STEP3_CONVERSION_SUMMARY.py

# Or try specific conversion
python convert_onnx_to_tflite_advanced.py
```

## Model Info
```
Input:   [1, 3, 512, 512] (RGB image, 512×512)
Outputs: 3×[1, 19, 512, 512] (segmentation masks)
Classes: 19 face parts (skin, eye, mouth, etc.)
```

## Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| "Module not found: onnx_tf" | `pip install onnx-tf` |
| "Long path error on Windows" | Use Linux/WSL or enable long paths |
| "Memory error" | Use machine with 16GB+ RAM |
| "Custom op: EagerPyFunc" | Don't use this method; use SavedModel instead |

## File Locations
```
Converting TO:  C:\Users\User\AndroidStudioProjects\mediapipe_inter_ver7\face-parsing.PyTorch\
Model SOURCE:   face_parsing.onnx
Model TARGET:   face_parsing.tflite (after conversion)
Python venv:    venv\ (already activated)
```

## What's Next?

1. **Do conversion** (pick Option A, B, or C above)
2. **Test model** using script in README_CONVERSION.md
3. **Deploy** to mobile/edge device
4. **Optimize** with quantization if needed

## Help Resources
- Full guide: `README_CONVERSION.md`
- Issues: `STEP3_ONNX_TO_TFLITE.md`
- Summary: `STEP3_CONVERSION_SUMMARY.py` (run it!)

## Key Facts
- ONNX: 111 KB
- TFLite target: 50-100 KB (with quantization)
- Inference: ~50-100ms mobile CPU
- Deployment: iOS, Android, Web, Edge

---
**Pro Tip**: Option A (SavedModel → TFLite) is most reliable!
Run: `python STEP3_CONVERSION_SUMMARY.py` for detailed walkthrough.
