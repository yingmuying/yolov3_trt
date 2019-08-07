
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
file(GLOB CXX_SRCS *.cpp)
file(GLOB CU_SRCS *.cu)

find_package(PkgConfig)
pkg_check_modules(GLIB REQUIRED glib-2.0)

include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${TRT_INCLUDE_DIR} ${GLIB_INCLUDE_DIRS} ${GLFAGS_INCLUDE_DIRS} )
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

CUDA_COMPILE(CU_OBJS ${CU_SRCS})
add_library(yolo-lib ${CXX_SRCS} ${CU_OBJS})

target_link_libraries(yolo-lib cudart cudnn cublas ${OpenCV_LIBRARIES} ${TRT_LIBRARY_INFER} ${TRT_LIBRARY_INFER_PLUGIN} gflags stdc++fs dl)