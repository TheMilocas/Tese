import onnxruntime as ort
import numpy as np
import cv2
from datasets import load_dataset
from PIL import Image
import onnx

dataset = load_dataset("Milocas/celebahq_masked", split="train")  

img_pil = dataset[0]["image"]
img = np.array(img_pil)
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
img = img.astype(np.float32)
img = (img - 127.5) / 128.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0) 

print(img.shape)  # (1, 3, 112, 112) → Ready for ArcFace ONNX

model_path = "ARCFACE/models/glintr100.onnx"
model = onnx.load(model_path)

target_output_name = "1330"

intermediate_value_info = onnx.helper.ValueInfoProto()
intermediate_value_info.name = target_output_name
model.graph.output.append(intermediate_value_info)

onnx.save(model, "ARCFACE/models/glintr100_modified.onnx")

session = ort.InferenceSession("ARCFACE/models/glintr100_modified.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
features = session.run([target_output_name], {input_name: img})[0]

print("Feature shape:", features.shape)  # Expected: (1, 512, 32, 32)
