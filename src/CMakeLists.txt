cmake_minimum_required(VERSION 3.8)
project(TensorRTApp)

set(TENSORRT_DIR "TensorRT-10.6.0.26")

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES "folder_pass.cpp")

include_directories(${CMAKE_SOURCE_DIR})

include_directories(${TENSORRT_DIR}/include)

link_directories(${TENSORRT_DIR}/lib)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} /usr/include)


add_executable(TensorRTApp ${SOURCE_FILES} "folder_pass.cpp")
target_link_libraries(TensorRTApp nvinfer_10 nvonnxparser_10 ${CUDA_LIBRARIES})
