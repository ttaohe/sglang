cmake_minimum_required(VERSION 3.18)
project(smid_recorder LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(pybind11_DIR /opt/conda/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 CONFIG REQUIRED)

set(Torch_DIR /opt/conda/lib/python3.10/site-packages/torch/share/cmake/Torch/)
find_package(Torch REQUIRED)

add_library(smid_recorder MODULE record_smid.cu)
set_target_properties(smid_recorder PROPERTIES PREFIX "")
set_property(TARGET smid_recorder PROPERTY CUDA_ARCHITECTURES 89)


target_link_libraries(smid_recorder PRIVATE pybind11::module ${TORCH_LIBRARIES})