cmake_minimum_required(VERSION 3.24)
project(cuda_state_vector LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
add_executable(cuda_state_vector src/main.cu)
target_link_libraries(cuda_state_vector PRIVATE CUDA::cudart)
set_target_properties(cuda_state_vector PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
