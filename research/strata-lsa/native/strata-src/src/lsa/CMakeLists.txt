cmake_minimum_required(VERSION 3.18)
project(strata_lsa_research LANGUAGES CXX)

# Standalone research targets. The production engine's root CMake project does
# not include this directory; enabling this build does not alter engine defaults.
get_filename_component(STRATA_LSA_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

option(STRATA_LSA_BUILD_CUDA "Build the standalone native CUDA LSA adapter" OFF)
set(STRATA_LSA_NATIVE_MAP "" CACHE FILEPATH "Optional exported pilot map for the native loader test")

add_library(strata_lsa STATIC maps.cpp cache_state.cpp)
target_include_directories(strata_lsa PUBLIC "${STRATA_LSA_SOURCE_ROOT}/include")
target_compile_features(strata_lsa PUBLIC cxx_std_17)
set_target_properties(strata_lsa PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(CTest)
if(BUILD_TESTING)
    add_executable(lsa_maps_test maps_test.cpp)
    target_link_libraries(lsa_maps_test PRIVATE strata_lsa)
    set(lsa_maps_test_args "${CMAKE_CURRENT_BINARY_DIR}/lsa-maps-fixture.bin")
    if(STRATA_LSA_NATIVE_MAP)
        if(NOT EXISTS "${STRATA_LSA_NATIVE_MAP}")
            message(FATAL_ERROR "STRATA_LSA_NATIVE_MAP does not exist: ${STRATA_LSA_NATIVE_MAP}")
        endif()
        list(APPEND lsa_maps_test_args "${STRATA_LSA_NATIVE_MAP}")
    endif()
    add_test(NAME lsa_maps COMMAND lsa_maps_test ${lsa_maps_test_args})

    add_executable(lsa_cache_state_test cache_state_test.cpp)
    target_link_libraries(lsa_cache_state_test PRIVATE strata_lsa)
    add_test(NAME lsa_cache_state COMMAND lsa_cache_state_test)
endif()

if(STRATA_LSA_BUILD_CUDA)
    # Set before language initialization. Callers can override this cache entry
    # for another explicitly qualified target; Onyx's device is SM120.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures for the isolated adapter")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(strata_lsa_cuda SHARED
        cuda_adapter.cu
        ward.cu
        "${STRATA_LSA_SOURCE_ROOT}/src/kernels/cuda/qsa_decode_attn.cu")
    target_include_directories(strata_lsa_cuda PRIVATE "${STRATA_LSA_SOURCE_ROOT}/include")
    # CUDA 13 packages CCCL below include/cccl; other toolkits expose cub
    # directly. Add the nested include only when that layout exists.
    foreach(cuda_include IN LISTS CUDAToolkit_INCLUDE_DIRS)
        if(EXISTS "${cuda_include}/cccl/cub/device/device_segmented_radix_sort.cuh")
            target_include_directories(strata_lsa_cuda PRIVATE "${cuda_include}/cccl")
        endif()
    endforeach()
    target_link_libraries(strata_lsa_cuda PRIVATE strata_lsa CUDA::cudart)
    set_target_properties(strata_lsa_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON)
endif()
