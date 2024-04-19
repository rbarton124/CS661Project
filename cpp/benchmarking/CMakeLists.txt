cmake_minimum_required(VERSION 3.2)

project(torch_cpp_benchmark)

# SET THIS TO WHERE YOUR LIBTORCH IS 
set(CMAKE_PREFIX_PATH /home/cw541/661/project/libtorch)

find_package(Torch REQUIRED)

list(APPEND CMAKE_CXX_FLAGS "${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} src/main.cpp 
                               src/cifar10.cpp 
                               src/common.cpp 
                               src/test.cpp
                               src/code_testing.cpp 
                               src/network.cpp 
                               )

target_include_directories(${PROJECT_NAME} PRIVATE include)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)