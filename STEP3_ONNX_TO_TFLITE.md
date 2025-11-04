# Step 3: ONNX → TensorFlow → TFLite Conversion Guide

## Project Status

✓ **Completed Steps:**
- Step 1: PyTorch → ONNX (via `export_onnx.py`)
  - Generated: `face_parsing.onnx` (0.11 MB)
- Step 2: ONNX Inspection (via `inspect_onnx_io.py`)
  - Verified model structure and I/O

⚠ **Current Step:**
- Step 3: ONNX → TensorFlow → TFLite

## Model Information

```
Model: Face Parsing Network
Input:  face_parsing.onnx
  - Shape: [1, 3, 512, 512]
  - Type: Float32
  
Outputs: 3 feature maps
  - output: [1, 19, 512, 512]
  - upsample_bilinear2d_1: [1, 19, 512, 512]
  - upsample_bilinear2d_2: [1, 19, 512, 512]
```

## Challenges with Direct ONNX → TFLite Conversion

The direct conversion from ONNX to TFLite is problematic because:

1. **Operator Incompatibility**: Not all ONNX operators map directly to TFLite ops
2. **Custom Operations**: Bilinear interpolation and other custom ops need special handling
3. **Python Wrapping**: Using `tf.py_function` creates custom ops that TFLite doesn't support

## Recommended Conversion Workflows

### Option 1: ONNX → SavedModel → TFLite (Best for Compatibility)

```bash
# Install required tools
pip install onnx-tf tensorflow-probability[tf]

# Convert ONNX to SavedModel
python
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("face_parsing.onnx")
tf_rep = prepare(onnx_model, strict=False)
tf_rep.export_graph("face_parsing_saved_model")

# Then convert SavedModel to TFLite
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model("face_parsing_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("face_parsing.tflite", "wb") as f:
    f.write(tflite_model)
```

### Option 2: PyTorch Direct Export (Simplest)

Export directly from PyTorch to TFLite, bypassing ONNX intermediate:

```python
import torch
import torch.onnx

# Instead of ONNX, export to SavedModel format
# Then convert to TFLite
```

### Option 3: Use MediaPipe Model Converter

For face parsing models, MediaPipe provides optimized converters:

```bash
pip install mediapipe-model-converter

# Convert directly with optimizations for mobile
```

### Option 4: Manual ONNX → TensorFlow SavedModel

```python
import onnx
import tensorflow as tf
from onnx import numpy_helper
import numpy as np

# 1. Load ONNX model
onnx_model = onnx.load("face_parsing.onnx")

# 2. Use ONNX Runtime for inference
import onnxruntime as rt
sess = rt.InferenceSession("face_parsing.onnx")

# 3. Create a TensorFlow SavedModel that wraps ONNX Runtime
class ONNXModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.sess = rt.InferenceSession("face_parsing.onnx")
        
    @tf.function(input_signature=[tf.TensorSpec([1, 3, 512, 512], tf.float32)])
    def __call__(self, inputs):
        # This won't work directly in TFLite due to py_function
        # But can be used for serving or optimization
        pass

# 4. Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()
```

## Current Status

**Generated Files:**
- ✓ `face_parsing.onnx` (0.11 MB) - from export_onnx.py
- ⚠ `face_parsing.tflite` (0.0 MB) - wrapper only, not functional

## Installation Issues Encountered

1. **Long Path Issue on Windows**: TensorFlow has issues with long file paths
   - Solution: Enable long paths in Windows or use shorter installation directory

2. **tf-keras Dependencies**: onnx-tf requires tf-keras for full functionality
   - Solution: `pip install tensorflow-probability[tf]`

3. **Custom Op Limitations**: TFLite can't handle Python-wrapped custom operations
   - Solution: Use proper ONNX → SavedModel conversion first

## Next Steps - Recommended Action

### For Development/Testing:
```bash
# Use ONNX Runtime directly for inference
pip install onnxruntime
# Then run inference directly without converting to TFLite
```

### For Mobile Deployment:
```bash
# Option A: Fix Windows long path issue and use onnx-tf
# Option B: Use pre-built MediaPipe solutions for face parsing
# Option C: Convert model on a Linux machine (no long path issues)
```

### For Production:
```bash
# 1. Create SavedModel from ONNX
# 2. Quantize with representative dataset
# 3. Test on target device
# 4. Deploy TFLite model
```

## TFLite Model Characteristics

When successfully converted, your TFLite model will:
- **Size**: ~100-200 KB (after quantization)
- **Latency**: ~50-100ms on mobile devices (CPU)
- **Memory**: ~10-50 MB runtime RAM

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Long path error | Windows path limits | Enable long paths or shorten paths |
| Module not found | Missing dependencies | Run `pip install` for required packages |
| Custom op error | EagerPyFunc in TFLite | Use proper ONNX → SavedModel conversion |
| Shape mismatch | Input dimension issue | Verify input shape [1, 3, 512, 512] |
| Memory error | Insufficient RAM | Reduce batch size or dataset size |

## Resources

- [TensorFlow Lite Conversion Guide](https://www.tensorflow.org/lite/convert)
- [ONNX to TensorFlow](https://github.com/onnx/onnx-tensorflow)
- [MediaPipe Model Maker](https://developers.google.com/mediapipe/solutions/customize)
- [TFLite Quantization](https://www.tensorflow.org/lite/performance/quantization)

## Testing the TFLite Model

Once converted, test with:

```python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter("face_parsing.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
test_input = np.random.rand(1, 3, 512, 512).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()

# Get outputs
for output in output_details:
    result = interpreter.get_tensor(output['index'])
    print(f"Output shape: {result.shape}")
```

---

**Last Updated**: 2025-11-04
**Status**: In Progress - ONNX model ready for conversion
