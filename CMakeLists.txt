cmake_minimum_required(VERSION 3.10)
if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "")
endif ()

project(CudaPathTracer)
enable_language(CUDA)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_package(OpenGL REQUIRED)
link_directories("${CMAKE_SOURCE_DIR}/lib")
include_directories( ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "${PROJECT_SOURCE_DIR}/src")

file(GLOB_RECURSE SOURCES "main.cpp"
    "src/*.cpp" "src/*.h" "src/*.cu" "src/*.cuh"
    "include/Tracer/*.cpp" "include/Tracer/*.h" "include/Tracer/*.cu" "include/Tracer/*.cuh"
)

set(CUDA_NVCC_COMPILE_DEFINITIONS "${CUDA_NVCC_COMPILE_DEFINITIONS}" "GLM_FORCE_PURE")
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
target_include_directories(${PROJECT_NAME} PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src/ImGui"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL $cudart)

if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE "assimp-vc140-mt" "FreeImage" "glew32" "glfw3dll")
    file(GLOB DLLS "${PROJECT_SOURCE_DIR}/dll/${LIB_DIR}/*.dll")
    file(COPY ${DLLS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

add_custom_command(
	TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink
		"${PROJECT_SOURCE_DIR}/Models"
		"${CMAKE_CURRENT_BINARY_DIR}/Models"
	DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/Models"
)
