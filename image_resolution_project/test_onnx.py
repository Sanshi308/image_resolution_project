import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("model.onnx")

# Create dummy input (1 image, 3 channels, 32x32)
dummy_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
input_name = session.get_inputs()[0].name

# Run inference
output = session.run(None, {input_name: dummy_input})
print("SUCCESS! Output shape:", output[0].shape)