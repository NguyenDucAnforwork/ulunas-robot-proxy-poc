cmake_minimum_required(VERSION 3.16)
project(ulunas_android_ref CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ORT_ROOT)
    set(ORT_ROOT "/content/project/mobile/ort_lib/extracted")
endif()

add_executable(ulunas_ref src/main.cpp)
target_include_directories(ulunas_ref PRIVATE ${ORT_ROOT}/headers)

if(ANDROID_ABI)
    target_link_libraries(ulunas_ref PRIVATE ${ORT_ROOT}/jni/${ANDROID_ABI}/libonnxruntime.so)
else()
    # x86_64 host build for fast iteration -- needs a host-arch onnxruntime .so, set via ORT_HOST_LIB
    if(NOT DEFINED ORT_HOST_LIB)
        message(FATAL_ERROR "set -DORT_HOST_LIB=/path/to/libonnxruntime.so for host builds")
    endif()
    target_link_libraries(ulunas_ref PRIVATE ${ORT_HOST_LIB})
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(ulunas_ref PRIVATE -O2)
endif()
