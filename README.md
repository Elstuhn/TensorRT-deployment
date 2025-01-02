# TensorRT-deployment
A deployment script in c++ for a modified deeplab-v3 inception net (segmentation)

### Compatibility
**This script can be used for any onnx or engine file that fulfills the below compatibilities:**
- Segmentation model (deserialization and conversion from onnx to engine is universal but pre/post processing was made specifically for segmentation)
- TensorRT v10.6 (code was created with TensorRT v10.6, there may be compatibility issues with certain code if other versions are used e.g EnqueueV3 etc)

### Pre/post Processing
**Input**<br/>
- Input images are **grayscaled then normalized** before being fed into engine, same with masks for comparison to calculate mIOU.
- If your engine is trained on and expects rgb channels, feel free to modify the code
<br/>
**Outputs**<br/>
- Output masks produced by engine during inference are expected to have same tensor shape as input images with values ranging [0, 1]
- Output is thresholded with default value 0.5 (can be adjusted to your needs)

## Performance
![image](https://github.com/user-attachments/assets/68987090-d4e4-47b8-8b43-f41160e06cd4)
Example of output after inference on image and mask folders

![image](https://github.com/user-attachments/assets/6157d188-0d8e-4df5-af8b-59f7f07d774c)
Example of image comparisons between ground truth and TensorRT(FP32) C++ with reference to original input image

<br/>

| Mode of Deployment  | Avg. Inference Time | Performance |
| ------------- | ------------- | ------------- |
| Pth file on python  | Undisclosed  | Undisclosed |
| ONNX on Python  | +3.77ms  | Unchanged |
| TensorRT (FP32) C++  | +3.31ms  | -0.01% |
| TensorRT (FP16) C++  | **-1.06ms**  | -0.01% |

Speed/performance comparisons from after pth file onwards are with reference to pth file performances (e.g unchanged = speed/performance same as pth file on python)

### Notes:
- ONNX model and imagepath not provided since it's confidential
- If TensorRT engine is not found, the engine will be built using the provided onnx path
- TensorRT version 10.6.0.26 was used (adjust TensorRT directory based on your versions for compatibility with your engine)
