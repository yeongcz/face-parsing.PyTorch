import onnx

model = onnx.load("face_parsing.onnx")
print("Inputs:")
for input in model.graph.input:
    print(f"  {input.name}")
print("Outputs:")
for output in model.graph.output:
    print(f"  {output.name}")
