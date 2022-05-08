import torch
import onnx
import torchvision

if __name__ == '__main__':
    model = torchvision.models.resnet18(True, True)
    model.eval()
    input = torch.randn(1, 3, 224, 224)
    f='resnet18.onnx'
    with torch.no_grad():
        output = model(input)
        print(output.size())
        torch.onnx.export(model,input,f,input_names=['image'],output_names=['output'])
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
