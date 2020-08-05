cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(demonet)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

add_executable(demonet detection.cpp)
target_link_libraries(demonet ${TORCH_LIBRARIES} TorchVision::TorchVision)

# set C++14 to compile PyTorch
set_property(TARGET demonet PROPERTY CXX_STANDARD 14)
