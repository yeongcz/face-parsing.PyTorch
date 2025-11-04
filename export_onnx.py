import torch
from model import BiSeNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, required=True, help='Path to the PyTorch checkpoint (.pth)')
parser.add_argument('--output', type=str, required=True, help='Output ONNX filename')
parser.add_argument('--input-size', type=int, nargs=2, default=[512, 512], help='Input size (H W)')
args = parser.parse_args()

n_classes = 19
model = BiSeNet(n_classes=n_classes)
model.eval()

# Load checkpoint
state_dict = torch.load(args.model_path, map_location='cpu')
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
model.load_state_dict(state_dict)

# Dummy input
dummy_input = torch.randn(1, 3, args.input_size[0], args.input_size[1])

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    args.output,
    input_names=['input'],
    output_names=['output'],
    opset_version=11,
    do_constant_folding=True,
    verbose=True
)
print(f"Exported ONNX model to {args.output}")
