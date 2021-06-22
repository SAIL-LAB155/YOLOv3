from models import *
import torch
import onnx

model = Darknet("/media/hkuit164/WD20EJRX/yolov3-channel-and-layer-pruning/cfg/yolov3-original-1cls-leaky.cfg")
model.load_weights("/media/hkuit164/WD20EJRX/yolov3-channel-and-layer-pruning/weights/rgb_146/best.weight")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

dummy_input = torch.randn(1, 3, 608, 608, device="cuda")

# preds = model(dummy_input)

input_names = ["input1"]
output_names = ["bboxes", "classes"]

torch.onnx.export(
    model,
    dummy_input,
    "./rgb146.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names)



model = onnx.load("models/rgb146.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))
