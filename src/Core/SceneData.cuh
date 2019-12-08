#pragma once

#define GLM_FORCE_PURE
#include <glm/glm.hpp>
#include <cmath>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "CUDA/CudaAssert.h"
#include "Core/Material.cuh"
#include "Core/Microfacet.cuh"

class MBVHNode;
struct BVHNode;
using namespace glm;

struct SceneData {
	bool skyboxEnabled = false;
	bool shadow = true;
	bool indirect = true;
	int skyboxTexture = -1;

	vec3* vertices = nullptr;
	vec3* normals = nullptr;
	vec3* centerNormals = nullptr;
	vec2* texCoords = nullptr;
	unsigned int* gpuMatIdxs = nullptr;
	uvec3* indices = nullptr;

	unsigned int* lightIndices = nullptr;
	unsigned int* gpuPrimIndices = nullptr;

	MBVHNode* gpuMbvhNodes = nullptr;
	BVHNode* gpuBvhNodes = nullptr;
	Material* gpuMaterials = nullptr;

	vec4* currentFrame = nullptr;
	unsigned int lightCount;
	unsigned int* gpuTexDims = nullptr;
	unsigned int* gpuTexOffsets = nullptr;
	vec4* gpuTexBuffer = nullptr;
	float normalEpsilon = 1e-5f;
	float distEpsilon = 1e-5f;
	float triangleEpsilon = 1e-3f;

	__host__ void deallocate()
	{
		cuda(Free(vertices));
		cuda(Free(normals));
		cuda(Free(centerNormals));
		cuda(Free(texCoords));
		cuda(Free(gpuMatIdxs));
		cuda(Free(indices));
		cuda(Free(lightIndices));
		cuda(Free(gpuPrimIndices));
		cuda(Free(gpuMbvhNodes));
		cuda(Free(gpuBvhNodes));
		cuda(Free(gpuMaterials));
		cuda(Free(currentFrame));
		cuda(Free(gpuTexDims));
		cuda(Free(gpuTexOffsets));
		cuda(Free(gpuTexBuffer));
	}

	__device__ inline Material &getMaterial(unsigned int hit_idx)
	{
		return gpuMaterials[gpuMatIdxs[hit_idx]];
	}

	__device__ inline unsigned int& getTextureWidth(unsigned int idx)
	{
		return gpuTexDims[idx];
	}

	__device__ inline unsigned int& getTextureHeight(unsigned int idx)
	{
		return gpuTexDims[idx + 1];
	}

	__device__ inline vec4& getTextureColor(unsigned int idx, vec2 texCoords)
	{
		const unsigned int width = gpuTexDims[idx * 2];
		const unsigned int height = gpuTexDims[idx * 2 + 1];

		float x = fmod(texCoords.x, 1.0f);
		float y = fmod(texCoords.y, 1.0f);

		if (x < 0) x += 1.0f;
		if (y < 0) y += 1.0f;

		const uint ix = uint(x * (width - 1));
		const uint iy = uint(y * (height - 1));

		const uint offset = gpuTexOffsets[idx] + (ix + iy * width);
		return gpuTexBuffer[offset];
	}

	__device__ inline vec3 getTextureNormal(unsigned int idx, vec2 texCoords)
	{
		const uint &width = gpuTexDims[idx * 2];
		const uint &height = gpuTexDims[idx * 2 + 1];

		float x = texCoords.x;
		float y = texCoords.y;

		if (x < 0) x += 1.0f;
		if (y < 0) y += 1.0f;

		const uint ix = uint(x * (width - 1));
		const uint iy = uint(y * (height - 1));

		const uint offset = gpuTexOffsets[idx] + (ix + iy * width);
		return normalize(vec3(gpuTexBuffer[offset].r, gpuTexBuffer[offset].g, gpuTexBuffer[offset].b));
	}

	__device__ inline float getTextureMask(unsigned int idx, vec2 texCoords)
	{
		const uint &width = gpuTexDims[idx * 2];
		const uint &height = gpuTexDims[idx * 2 + 1];

		float x = texCoords.x;
		float y = texCoords.y;

		if (x < 0) x += 1.0f;
		if (y < 0) y += 1.0f;

		const uint ix = uint(x * (width - 1));
		const uint iy = uint(y * (height - 1));

		const uint offset = gpuTexOffsets[idx] + (ix + iy * width);
		return gpuTexBuffer[offset].r;
	}
};

#ifdef __CUDACC__
__device__ inline int atomicAggInc(int *ctr) {
	using namespace cooperative_groups;

	auto g = coalesced_threads();
	int warp_res;
	if (g.thread_rank() == 0)
		warp_res = atomicAdd(ctr, g.size());
	return g.shfl(warp_res, 0) + g.thread_rank();
}
#endif