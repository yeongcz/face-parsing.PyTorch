#!/usr/bin/env python3
"""
Simple ONNX to TFLite Conversion
Uses the existing face_parsing.onnx file
"""

import os
import sys
import subprocess

print("="*70)
print("ONNX → TFLite Conversion")
print("="*70)

# Check for ONNX file
onnx_file = "face_parsing.onnx"
if not os.path.exists(onnx_file):
    print(f"\n✗ Error: {onnx_file} not found!")
    print("Please ensure the ONNX model is in the current directory.")
    sys.exit(1)

print(f"\n✓ Found ONNX model: {onnx_file}")
size_kb = os.path.getsize(onnx_file) / 1024
print(f"  Size: {size_kb:.2f} KB")

# Method: Use onnx2tf tool
print("\n[Step 1] Installing onnx2tf...")
try:
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "onnx2tf", "-q"
    ], check=True)
    print("✓ onnx2tf installed")
except:
    print("✓ onnx2tf already installed or installation skipped")

# Convert using onnx2tf
print("\n[Step 2] Converting ONNX to TFLite...")
try:
    import onnx2tf
    
    # Convert
    onnx2tf.convert(
        input_onnx_file_path=onnx_file,
        output_folder_path="tflite_output",
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True
    )
    
    # Find generated TFLite file
    tflite_files = []
    if os.path.exists("tflite_output"):
        for f in os.listdir("tflite_output"):
            if f.endswith(".tflite"):
                tflite_files.append(os.path.join("tflite_output", f))
    
    if tflite_files:
        # Copy to main directory
        import shutil
        main_tflite = "face_parsing.tflite"
        shutil.copy(tflite_files[0], main_tflite)
        
        size_kb = os.path.getsize(main_tflite) / 1024
        print(f"\n✓ TFLite model created: {main_tflite}")
        print(f"  Size: {size_kb:.2f} KB")
        
        print("\n" + "="*70)
        print("CONVERSION COMPLETE!")
        print("="*70)
        print(f"\n✓ Generated: {main_tflite}")
        print(f"  Ready for deployment on mobile devices!")
        
    else:
        print("\n⚠ TFLite file not found in expected location")
        print("  Check tflite_output folder for generated files")
        
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    print("\nTrying alternative method...")
    
    # Alternative: Use command line
    print("\n[Alternative] Using onnx2tf command line...")
    try:
        result = subprocess.run([
            "onnx2tf",
            "-i", onnx_file,
            "-o", "tflite_output"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Conversion successful")
            
            # Find TFLite file
            import glob
            tflite_files = glob.glob("tflite_output/*.tflite")
            if tflite_files:
                import shutil
                main_tflite = "face_parsing.tflite"
                shutil.copy(tflite_files[0], main_tflite)
                
                size_kb = os.path.getsize(main_tflite) / 1024
                print(f"\n✓ TFLite model: {main_tflite}")
                print(f"  Size: {size_kb:.2f} KB")
        else:
            print(f"✗ Command failed: {result.stderr}")
            
    except FileNotFoundError:
        print("✗ onnx2tf command not found")
        print("\nManual steps:")
        print("  1. pip install onnx2tf")
        print("  2. onnx2tf -i face_parsing.onnx -o tflite_output")
        
except Exception as e:
    print(f"\n✗ Conversion failed: {e}")
    import traceback
    traceback.print_exc()
