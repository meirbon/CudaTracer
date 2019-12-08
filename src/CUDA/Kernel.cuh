#pragma once

#include <GL/glew.h>

#include <cuda_runtime.h>

#define GLM_FORCE_PURE
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Core/Camera.h"
#include "Core/Ray.cuh"
#include "Core/Triangle.cuh"
#include "BVH/MBVHNode.cuh"
#include "Core/SceneData.cuh"
#include "CUDA/CudaAssert.h"

using namespace core;
using namespace glm;
using namespace triangle;

struct Params {
	int width;
	int height;
	int samples;
	int smCores;

	Camera* camera = nullptr;
	SceneData gpuScene;

	Ray* gpuRays = nullptr;
	Ray* gpuNextRays = nullptr;
	ShadowRay* gpuShadowRays = nullptr;

	bool reference = false;

	Params(int width, int height)
	{
		this->width = width;
		this->height = height;
		this->samples = 0;

		int glDeviceId;
		unsigned int glDeviceCount;
		cudaGLGetDevices(&glDeviceCount, &glDeviceId, 1u, cudaGLDeviceListAll);

		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, glDeviceId);
		this->smCores = props.multiProcessorCount;

		std::cout << props.name << ", SM Count: " << smCores << std::endl;

		gpuScene.skyboxEnabled = false;
		gpuScene.skyboxTexture = -1;
	}

	__host__ inline void deallocate()
	{
		gpuScene.deallocate();
		cuda(Free(gpuRays));
		cuda(Free(gpuNextRays));
		cuda(Free(gpuShadowRays));
	}

	__host__ inline void reset()
	{
		cudaMemset(gpuScene.currentFrame, 0, width * height * sizeof(vec4));
		cudaDeviceSynchronize();
		samples = 0;
	}
};

cudaError_t launchKernels(cudaArray_const_t array, Params& params, int rayBufferSize);