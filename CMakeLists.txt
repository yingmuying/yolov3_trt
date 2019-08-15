cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(trt-yolo-app LANGUAGES CXX)

add_definitions(-fpermissive -fPIC)   # 预编译选项

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wunused-function -Wunused-variable -Wfatal-errors")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

set(TRT_SDK_ROOT "/home/***/TensorRT-5.1.5.0/" CACHE PATH "NVIDIA TensorRT SDK root path")
set(OPENCV_ROOT "" CACHE PATH "OpenCV SDK root path")

find_package(CUDA 9.0 EXACT REQUIRED cudart cublas curand)
list(APPEND GPU_ARCHS 30 35 37 50 52 60 61 70 75)


# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
  set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

# Generate PTX for the last architecture
list(GET GPU_ARCHS -1 LATEST_GPU_ARCH)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_GPU_ARCH},code=compute_${LATEST_GPU_ARCH}")

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find TensorRT
find_path(TRT_INCLUDE_DIR NvInfer.h HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES include)
if(${TRT_INCLUDE_DIR} MATCHES "TRT_INCLUDE_DIR-NOTFOUND")
  MESSAGE(FATAL_ERROR "-- Unable to find TensorRT headers. Please set path using -DTRT_SDK_ROOT")
else()
  MESSAGE(STATUS "Found TensorRT headers at ${TRT_INCLUDE_DIR}")
endif()


# TRT_LIBRARY TRT_INCLUDE_DIR
find_library(TRT_LIBRARY_INFER nvinfer HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_LIBRARY_INFER_PLUGIN nvinfer_plugin HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
if((${TRT_LIBRARY_INFER} MATCHES "TRT_LIBRARY_INFER-NOTFOUND") OR (${TRT_LIBRARY_INFER_PLUGIN} MATCHES "TRT_LIBRARY_INFER_PLUGIN-NOTFOUND"))
  MESSAGE(FATAL_ERROR "-- Unable to find TensorRT libs. Please set path using -DTRT_SDK_ROOT")
else()
  set(TRT_LIBRARY ${TRT_LIBRARY_INFER} ${TRT_LIBRARY_INFER_PLUGIN})
  MESSAGE(STATUS "Found TensorRT libs at ${TRT_LIBRARY}")
endif()

# Add yolo lib as subdir
add_subdirectory(${PROJECT_SOURCE_DIR}/lib ${PROJECT_BINARY_DIR}/lib)

include_directories(${CUDA_INCLUDE_DIRS} ${TRT_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/lib)    # -I
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)     #  -L

add_executable(trt-yolo-app trt-yolo-app.cpp)
target_link_libraries(trt-yolo-app yolo-lib)
