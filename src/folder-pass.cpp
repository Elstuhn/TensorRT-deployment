#include <filesystem>
#include <chrono>
#include <numeric>
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
#include <cmath>
#include <stdexcept>

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

class Int8EntropyCalibrator : public IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(const vector<string>& imagePaths, int inputWidth, int inputHeight, int inputChannels)
        : mImagePaths(imagePaths), mInputWidth(inputWidth), mInputHeight(inputHeight), mInputChannels(inputChannels), mCurrentIndex(0) {
        mInputCount = inputWidth * inputHeight * inputChannels;
        cudaMalloc(&mDeviceInput, mInputCount * sizeof(float));
    }

    ~Int8EntropyCalibrator() {
        cudaFree(mDeviceInput);
    }

    int getBatchSize() const noexcept override {
        return 1; // Batch size of 1 for simplicity
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        if (mCurrentIndex >= mImagePaths.size()) {
            return false; // No more data
        }

        const string& imagePath = mImagePaths[mCurrentIndex];
        vector<float> inputData = preprocessImage(imagePath, mInputWidth, mInputHeight, mInputChannels);

        cudaMemcpy(mDeviceInput, inputData.data(), mInputCount * sizeof(float), cudaMemcpyHostToDevice);
        bindings[0] = mDeviceInput;

        mCurrentIndex++;
        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept override {
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        mCalibrationCache.assign(reinterpret_cast<const char*>(cache), reinterpret_cast<const char*>(cache) + length);
    }

private:
    vector<string> mImagePaths;
    int mInputWidth, mInputHeight, mInputChannels;
    size_t mInputCount;
    size_t mCurrentIndex;
    void* mDeviceInput;
    vector<char> mCalibrationCache;

    vector<float> preprocessImage(const string& imagePath, int width, int height, int channels) {
        int w, h, c;
        unsigned char* img = stbi_load(imagePath.c_str(), &w, &h, &c, channels);
        if (!img) {
            cerr << "Failed to load image: " << imagePath << endl;
            throw runtime_error("Image loading failed");
        }

        vector<float> normalizedImage(w * h * c);
        for (int i = 0; i < w * h * c; ++i) {
            normalizedImage[i] = static_cast<float>(img[i]) / 255.0f;
        }
        stbi_image_free(img);

        return normalizedImage;
    }
};

float computeDefectIoU(const vector<unsigned char>& pred, const vector<unsigned char>& gt) {
    size_t intersection = 0;
    size_t unionArea = 0;

    for (size_t i = 0; i < pred.size(); ++i) {
        intersection += (pred[i] == 255 && gt[i] == 255);
        unionArea += (pred[i] == 255 || gt[i] == 255);
    }

    return unionArea > 0 ? static_cast<float>(intersection) / unionArea : 1.0f;
}

float computeBackgroundIoU(const vector<unsigned char>& pred, const vector<unsigned char>& gt) {
    size_t intersection = 0;
    size_t unionArea = 0;

    for (size_t i = 0; i < pred.size(); ++i) {
        intersection += (pred[i] == 0 && gt[i] == 0);
        unionArea += (pred[i] == 0 || gt[i] == 0);
    }

    return unionArea > 0 ? static_cast<float>(intersection) / unionArea : 1.0f;
}

ICudaEngine* buildEngineFromONNX(const string& onnxPath, IBuilder* builder, IBuilderConfig* config, bool enableINT8, const vector<string>& calibrationPaths) {
    auto network = builder->createNetworkV2(static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        cerr << "Failed to parse ONNX model: " << onnxPath << endl;
        delete parser;
        delete network;
        return nullptr;
    }

    std::cout << "ONNX model parsed successfully." << std::endl;

    if (enableINT8 && builder->platformHasFastInt8()) {
        std::cout << "INT8 supported, enabling INT8 precision." << std::endl;
        config->setFlag(BuilderFlag::kINT8);

        // Create calibrator
        auto calibrator = new Int8EntropyCalibrator(calibrationPaths, 512, 512, 1); // Adjust dimensions
        config->setInt8Calibrator(calibrator);
    }
    else if (builder->platformHasFastFp16()) {
        std::cout << "FP16 supported, enabling FP16 precision." << std::endl;
        config->setFlag(BuilderFlag::kFP16);
    }
    else {
        std::cout << "Defaulting to FP32 precision." << std::endl;
    }

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        cerr << "Failed to build TensorRT engine." << endl;
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
    std::cout << "TensorRT engine saved to " << enginePath << endl;
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
    const char* imagePath = "data/JJ-train/images/Adhesion_000_04_FlakingBlack_062b_clean_blend.png";
    const string imageFolder = "data/onsite/images";
    const string maskFolder = "data/onsite/masks_inv";
    int width, height, channels;
    const bool int8flag = false;
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

    const string onnxPath = "resnet50-07-02-2024_09-hrs.onnx";

    vector<string> calibrationPaths;
    for (const auto& entry : fs::directory_iterator(imageFolder)) {
        if (entry.path().extension() == ".png") {
            calibrationPaths.push_back(entry.path().string());
        }
    }

    if (calibrationPaths.empty()) {
        cerr << "No calibration images found in folder: " << imageFolder << endl;
        return -1;
    }

    if (!engineFile) {
        std::cout << "Engine file not found. Building from ONNX model..." << endl;
        auto builder = createInferBuilder(gLogger);
        auto config = builder->createBuilderConfig();
        //engine = buildEngineFromONNX(onnxPath, builder, config);
        ICudaEngine* engine = buildEngineFromONNX(onnxPath, builder, config, int8flag, calibrationPaths);
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
        std::cout << "Loading engine from file: " << enginePath << endl;
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

    float totalDefectIoU = 0.0f;
    float totalBackgroundIoU = 0.0f;
    int imageCount = 0;
    double totalInferenceTime = 0.0;
    for (const auto& entry : fs::directory_iterator(imageFolder)) {
        if (entry.path().extension() != ".png") continue;

        string imagePath = entry.path().string();
        string imageName = entry.path().stem().string();
        string maskPath = maskFolder + "/" + imageName + "_label.png";

        if (!fs::exists(maskPath)) {
            cerr << "Mask not found for image: " << imageName << endl;
            continue;
        }

        int width, height, channels;
        unsigned char* h_image = stbi_load(imagePath.c_str(), &width, &height, &channels, 1);
        if (!h_image) {
            cerr << "Failed to load image: " << imagePath << endl;
            continue;
        }

        vector<float> h_normalized_image(width * height);
        for (int i = 0; i < width * height; ++i) {
            h_normalized_image[i] = static_cast<float>(h_image[i]) / 255.0f;
        }

        float* d_input;
        float* d_output;
        size_t inputSize = width * height * sizeof(float);
        size_t outputSize = width * height * sizeof(float);

        cudaMalloc(&d_input, inputSize);
        cudaMalloc(&d_output, outputSize);
        cudaMemcpy(d_input, h_normalized_image.data(), inputSize, cudaMemcpyHostToDevice);


        context->setInputTensorAddress("input", d_input);
        context->setOutputTensorAddress("output", d_output);
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        auto start = chrono::high_resolution_clock::now();
        if (!context->enqueueV3(stream)) {
            cerr << "Failed to execute TensorRT inference." << endl;
            continue;
        }

        cudaStreamSynchronize(stream);
        auto end = chrono::high_resolution_clock::now();
     

        double inferenceTime = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        totalInferenceTime += inferenceTime;

        vector<float> hostOutput(width * height);
        cudaMemcpy(hostOutput.data(), d_output, outputSize, cudaMemcpyDeviceToHost);

        const float threshold = 0.5f;
        vector<unsigned char> binaryOutput(width * height);
        transform(hostOutput.begin(), hostOutput.end(), binaryOutput.begin(),
            [threshold](float probability) { return (probability > threshold) ? 255 : 0; });

        string predictedMaskPath = "./inference-results-fp32/" + imageName + "_predicted.png";
        if (!stbi_write_png(predictedMaskPath.c_str(), width, height, 1, binaryOutput.data(), width)) {
            cerr << "Failed to save predicted mask for: " << imageName << endl;
        }

        int maskWidth, maskHeight, maskChannels;
        unsigned char* h_mask = stbi_load(maskPath.c_str(), &maskWidth, &maskHeight, &maskChannels, 1);

        if (!h_mask || maskWidth != width || maskHeight != height) {
            cerr << "Invalid mask dimensions for: " << maskPath << endl;
            stbi_image_free(h_image);
            continue;
        }

        vector<unsigned char> groundTruth(h_mask, h_mask + width * height);
        float defectiou = computeDefectIoU(binaryOutput, groundTruth);
        totalDefectIoU += defectiou;
        float backgroundiou = computeBackgroundIoU(binaryOutput, groundTruth);
        totalBackgroundIoU += backgroundiou;

        stbi_image_free(h_image);
        stbi_image_free(h_mask);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);

        imageCount++;
    }

    float avgBackgroundIoU = totalBackgroundIoU / imageCount;
    float avgDefectIoU = totalDefectIoU / imageCount;
    float mIoU = (avgBackgroundIoU + avgDefectIoU) / 2;
    double avgInferenceTime = totalInferenceTime / imageCount;

    std::cout << "Processed " << imageCount << " images." << endl;
    std::cout << "mIoU: " << mIoU << endl;
    std::cout << "Average Background IoU: " << avgBackgroundIoU << endl;
    std::cout << "Average Defect IoU: " << avgDefectIoU << endl;
    std::cout << "Average Inference Time: " << avgInferenceTime << " ms" << endl;
    std::cout << "Total Inference Time: " << totalInferenceTime << " ms" << endl;

    delete context;
    delete engine;
    delete runtime;

    return 0;
}
