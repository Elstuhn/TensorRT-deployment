# TensorRT-deployment
A deployment script in c++ for a modified deeplab-v3 inception net

### Notes:
- ONNX model and imagepath not provided since it's confidential
- If TensorRT engine is not found, the engine will be built using the provided onnx path
- TensorRT version 10.6.0.26 was used


## Examples
![image](https://github.com/user-attachments/assets/6157d188-0d8e-4df5-af8b-59f7f07d774c)

| Mode of Deployment  | Avg. Inference Time | Performance |
| ------------- | ------------- | ------------- |
| Pth file on python  | Undisclosed  | Undisclosed |
| ONNX on Python  | +3.77ms  | Unchanged |
| TensorRT (FP32) C++  | +3.31ms  | -0.01% |
| TensorRT (FP16) C++  | **-1.06ms**  | -0.01% |

Speed/performance comparisons from after pth file onwards are with reference to pth file performances (e.g unchanged = speed/performance same as pth file on python)
