#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GLM_FORCE_PURE
#include <glm/glm.hpp>

#include "Core/Camera.h"
#include "Core/Ray.cuh"
#include "Triangle.cuh"
#include "BVH/MBVHNode.cuh"
#include "Core/SceneData.cuh"

using namespace core;
using namespace glm;
using namespace triangle;

struct Params {
	int width;
	int height;
	int smCores;
	Camera* camera;
	SceneData gpuScene;

	Ray* gpuRays = nullptr;
	Ray* gpuNextRays = nullptr;
	ShadowRay* gpuShadowRays = nullptr;
};

cudaError_t launchKernels(cudaArray_const_t array, Params& params, int samples, int rayBufferSize);