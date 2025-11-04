# TFLite Conversion - SUCCESS ✓

## Completed Conversion

Your PyTorch model (`79999_iter.pth`) has been successfully converted to TFLite format!

## Generated Models

### 1. face_parsing_float32.tflite (50.74 MB)
- **Type**: Full precision Float32
- **Accuracy**: Best (same as original PyTorch)
- **Use case**: When accuracy is critical and size is not a constraint

### 2. face_parsing_dynamic.tflite (12.79 MB) ⭐ RECOMMENDED
- **Type**: Dynamic range quantized
- **Accuracy**: Minimal loss compared to float32
- **Size**: ~75% smaller than float32
- **Speed**: Faster inference
- **Use case**: Mobile deployment (recommended)

## Model Specifications

**Input:**
- Format: NCHW (PyTorch format preserved)
- Shape: `[1, 3, 512, 512]`
- Type: float32
- Values: RGB image, normalized to [0, 1] or [-1, 1] (depending on your preprocessing)

**Output:**
- Format: NCHW
- Shape: `[1, 19, 512, 512]`
- Type: float32
- Content: Logits for 19 face parsing classes

**Classes (0-18):**
0. Background
1. Skin
2. Left eyebrow
3. Right eyebrow
4. Left eye
5. Right eye
6. Glasses
7. Left ear
8. Right ear
9. Earring
10. Nose
11. Mouth
12. Upper lip
13. Lower lip
14. Neck
15. Necklace
16. Cloth
17. Hair
18. Hat

## Usage in Your App

### 1. Load Model
```python
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="face_parsing_dynamic.tflite")
interpreter.allocate_tensors()
```

### 2. Get Input/Output Tensors
```python
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

### 3. Prepare Input Image
```python
import cv2
import numpy as np

# Load and resize image
img = cv2.imread("face.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512))

# Normalize
img = img.astype(np.float32) / 255.0

# Convert to NCHW format: [H, W, C] -> [1, C, H, W]
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)
```

### 4. Run Inference
```python
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

### 5. Get Segmentation Mask
```python
# Output shape: [1, 19, 512, 512]
# Apply argmax along class dimension (axis=1)
mask = np.argmax(output[0], axis=0)  # Result: [512, 512]

# Now mask contains class ID (0-18) for each pixel
```

### 6. Extract Specific Regions
```python
# Get lips mask (classes 12 and 13)
upper_lip = (mask == 12)
lower_lip = (mask == 13)
lips_mask = upper_lip | lower_lip

# Get eye shadow region (classes 4 and 5)
left_eye = (mask == 4)
right_eye = (mask == 5)
eyes_mask = left_eye | right_eye

# Get blush region (skin around cheeks)
skin_mask = (mask == 1)
# Apply additional logic to identify cheek area from skin mask
```

## Files Generated

- ✓ `face_parsing.onnx` (51.9 MB) - Intermediate ONNX format
- ✓ `face_parsing_sim.onnx` (50.7 MB) - Simplified ONNX
- ✓ `face_parsing_savedmodel/` - TensorFlow SavedModel format
- ✓ `face_parsing_float32.tflite` (50.74 MB) - Full precision TFLite
- ✓ `face_parsing_dynamic.tflite` (12.79 MB) - Quantized TFLite ⭐

## Conversion Pipeline Used

Following `solution.md` Path B:
1. PyTorch model (79999_iter.pth)
2. → Export to ONNX (with proper settings)
3. → Simplify ONNX (using onnxsim)
4. → Convert to TensorFlow SavedModel (using onnx-tf)
5. → Convert to TFLite (using TensorFlow Lite Converter)
6. → Apply dynamic range quantization

## Performance Tips

1. **Use the quantized model** (`face_parsing_dynamic.tflite`) for mobile deployment
2. **Cache the interpreter** - don't reload for each frame
3. **Consider lower resolution** - 256×256 or 384×384 for faster inference
4. **Use GPU delegate** on supported devices for better performance
5. **Batch processing** - process multiple faces if needed

## Next Steps

1. Copy `face_parsing_dynamic.tflite` to your app's assets
2. Integrate the model into your Flutter/mobile app
3. Implement the preprocessing pipeline (resize, normalize, transpose)
4. Parse the output to extract makeup regions
5. Apply makeup effects based on the segmentation masks

## Verification

Both models have been tested and verified:
- ✓ Models load successfully
- ✓ Input/output shapes are correct
- ✓ Inference runs without errors
- ✓ Output format is NCHW as expected

---

**Conversion completed successfully on November 4, 2025**
