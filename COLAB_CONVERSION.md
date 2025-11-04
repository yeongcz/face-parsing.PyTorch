# Face Parsing: ONNX to TFLite Conversion (Google Colab)

## Instructions:
## 1. Go to https://colab.research.google.com
## 2. Create a new notebook
## 3. Upload face_parsing_v2.onnx (from this directory)
## 4. Copy and paste the cells below

# Cell 1: Install dependencies
!pip install onnx2tf -q

# Cell 2: Convert ONNX to TFLite  
!onnx2tf -i face_parsing_v2.onnx -o tflite_output

# Cell 3: Check output
import os
for file in os.listdir('tflite_output'):
    if file.endswith('.tflite'):
        size_kb = os.path.getsize(f'tflite_output/{file}') / 1024
        print(f"✓ Generated: {file} ({size_kb:.2f} KB)")

# Cell 4: Download the TFLite model
from google.colab import files
# Find and download the .tflite file
import glob
tflite_files = glob.glob('tflite_output/*.tflite')
if tflite_files:
    files.download(tflite_files[0])
else:
    print("No TFLite file found")

## That's it! The face_parsing.tflite file will download to your computer
