import torch
from utils import SimpleSRCNN

def export_to_onnx():
    model = SimpleSRCNN(scale_factor=2)
    
    # Load trained weights if available
    try:
        model.load_state_dict(torch.load("sr_model.pth", map_location="cpu"))
        print("Loaded trained weights from sr_model.pth")
    except FileNotFoundError:
        print("sr_model.pth not found, exporting base architecture.")

    model.eval()

    # Dummy Input (Batch size 1, 3 channels, 32x32 image)
    dummy_input = torch.randn(1, 3, 32, 32)

    # Export to ONNX
    onnx_path = "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model successfully exported to {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()