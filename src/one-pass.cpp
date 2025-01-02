#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <algorithm>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <vector>
 
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;
namespace fs = std::filesystem;
 
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};
 
Logger gLogger;
 
float computeIoU(const vector<unsigned char>& pred, const vector<unsigned char>& gt) {
    size_t intersection = 0;
    size_t unionArea = 0;
 
    for (size_t i = 0; i < pred.size(); ++i) {
        intersection += (pred[i] == 255 && gt[i] == 255);
        unionArea += (pred[i] == 255 || gt[i] == 255);
    }
 
    return unionArea > 0 ? static_cast<float>(intersection) / unionArea : 1.0f;
}
 
ICudaEngine* buildEngineFromONNX(const string& onnxPath, IBuilder* builder, IBuilderConfig* config) {
    auto network = builder->createNetworkV2(static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
 
    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        cerr << "Failed to parse ONNX model: " << onnxPath << endl;
        delete parser;
        delete network;
        return nullptr;
    }
 
    std::cout << "ONNX model parsed successfully." << std::endl;
 
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        cerr << "Failed to build TensorRT engine from ONNX model." << endl;
    }
 
    delete parser;
    delete network;
    return engine;
}
 
void saveEngine(ICudaEngine* engine, const string& enginePath) {
    IHostMemory* serializedEngine = engine->serialize();
    if (!serializedEngine) {
        cerr << "Failed to serialize the TensorRT engine." << endl;
        return;
    }
    ofstream engineFile(enginePath, ios::binary);
    if (!engineFile) {
        cerr << "Failed to open file to save the TensorRT engine." << endl;
        delete serializedEngine;
        return;
    }
    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    delete serializedEngine;
    cout << "TensorRT engine saved to " << enginePath << endl;
}
 
ICudaEngine* loadEngine(const string& enginePath, IRuntime* runtime) {
    ifstream engineFile(enginePath, ios::binary);
    if (!engineFile) {
        cerr << "Failed to open engine file: " << enginePath << endl;
        return nullptr;
    }
 
    engineFile.seekg(0, ios::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, ios::beg);
 
    vector<char> engineData(fileSize);
    engineFile.read(engineData.data(), fileSize);
    engineFile.close();
 
    // Deserialize the engine using TensorRT runtime
    return runtime->deserializeCudaEngine(engineData.data(), fileSize);
}
 
int main() {
    const char* imagePath = "imagepath";
    int width, height, channels;
    const string enginePath = "deeplab-resnet50.engine";
 
    unsigned char* h_image = stbi_load(imagePath, &width, &height, &channels, 1);
    if (!h_image) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return -1;
    }
 
    std::cout << "Image loaded. Dimensions: " << width << "x" << height << std::endl;
 
    float* h_normalized_image = new float[1 * channels * height * width];
    for (int i = 0; i < width * height; ++i) {
        h_normalized_image[i] = static_cast<float>(h_image[i]) / 255.0f;
    }
 
    float* d_image;
    size_t imageSize = 1 * channels * height * width * sizeof(float);
    cudaMalloc(&d_image, imageSize);
 
    cudaMemcpy(d_image, h_normalized_image, imageSize, cudaMemcpyHostToDevice);
 
    std::cout << "Image uploaded to GPU memory." << std::endl;
 
    ICudaEngine* engine = nullptr;
    IRuntime* runtime = createInferRuntime(gLogger);
 
    ifstream engineFile(enginePath, ios::binary);
 
    const string onnxPath = "onnxfile";
 
    if (!engineFile) {
        cout << "Engine file not found. Building from ONNX model..." << endl;
        auto builder = createInferBuilder(gLogger);
        auto config = builder->createBuilderConfig();
        engine = buildEngineFromONNX(onnxPath, builder, config);
        if (!engine) {
            delete config;
            delete builder;
            return -1;
        }
        saveEngine(engine, enginePath); 
        delete config;
        delete builder;
    }
    else {
        cout << "Loading engine from file: " << enginePath << endl;
        engine = loadEngine(enginePath, runtime);
        if (!engine) {
            cerr << "Failed to load TensorRT engine from file." << endl;
            return -1;
        }
    }
 
    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        cerr << "Failed to create TensorRT execution context." << endl;
        delete engine;
        delete runtime;
        return -1;
    }
 
    float* d_input;
    float* d_output;
    size_t inputSize = 1 * channels * height * width * sizeof(float); //float_32 byte size
    size_t outputSize = 1 * 1 * height * width * sizeof(float);
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);
 
    cudaMemcpy(d_input, d_image, inputSize, cudaMemcpyDeviceToDevice);
 
    void* buffers[] = { d_input, d_output };
 
    cudaStream_t stream;
 
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream." << std::endl;
        return -1;
    }
 
    if (!context->setInputTensorAddress("input", d_input)) {
        cerr << "Failed to set input tensor address." << endl;
        cudaStreamDestroy(stream);
        return -1;
    }
    if (!context->setOutputTensorAddress("output", d_output)) {
        cerr << "Failed to set output tensor address." << endl;
        cudaStreamDestroy(stream);
        return -1;
    }
 
    if (!context->enqueueV3(stream)) {
        cerr << "Failed to execute TensorRT inference with enqueueV3." << endl;
        cudaStreamDestroy(stream);
        return -1;
    }
 
    cudaStreamSynchronize(stream);
 
    vector<float> hostOutput(width * height);
    cudaMemcpy(hostOutput.data(), d_output, outputSize, cudaMemcpyDeviceToHost);
 
    const float threshold = 0.5f;
 
    vector<unsigned char> binaryOutput(width * height);
    std::transform(hostOutput.begin(), hostOutput.end(), binaryOutput.begin(),
        [threshold](float probability) -> unsigned char {
            return (probability > threshold) ? 255 : 0;
        });
 
    const char* outputPath = "output_mask.png";
    if (stbi_write_png(outputPath, width, height, 1, binaryOutput.data(), width) == 0) {
        cerr << "Failed to save the binary mask as an image." << endl;
        return -1;
    }
 
    stbi_image_free(h_image);
    cudaFree(d_image);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;
    delete[] h_normalized_image;
 
    return 0;
}
