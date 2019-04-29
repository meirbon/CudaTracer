#pragma once

#include <cuda_runtime.h>
#include <atomic>
#include <immintrin.h>
#include <thread>
#include <vector>

#include "Utils/ctpl.h"
#include "BVH/AABB.h"
#include "Core/SceneData.cuh"
#include "Core/Triangle.cuh"
#include "Core/TriangleList.h"

class BVHTree;

constexpr float EPSILON_T = 0.001f;

struct BVHTraversal
{
	int nodeIdx;
	float tNear;

	__device__ __host__ BVHTraversal() {};
	__device__ __host__ inline BVHTraversal(int nIdx, float t)
		: nodeIdx(nIdx), tNear(t)
	{
	}
};

struct BVHNode
{
public:
	union {
		AABB bounds;
		struct {
			glm::vec3 min;
			int leftFirst;
			glm::vec3 max;
			int count;
		};
	};

	BVHNode();

	BVHNode(int leftFirst, int count, AABB bounds);

	~BVHNode() = default;

	__device__ __host__ inline bool IsLeaf() const noexcept { return bounds.count > -1; }

	__device__ __host__ inline bool Intersect(const glm::vec3& org, const glm::vec3& dirInverse, float* tmin, float* tmax) const
	{
		const vec3 t1 = (this->min - org) * dirInverse;
		const vec3 t2 = (this->max - org) * dirInverse;

		const vec3 min = glm::min(t1, t2);
		const vec3 max = glm::max(t1, t2);

		*tmin = glm::max(min.x, glm::max(min.y, min.z));
		*tmax = glm::min(max.x, glm::min(max.y, max.z));

		return *tmax >= 0.0f && *tmin < *tmax;
	}

	inline void SetCount(int value) noexcept { bounds.count = value; }

	inline void SetLeftFirst(unsigned int value) noexcept { bounds.leftFirst = value; }

	__device__ __host__ inline int GetCount() const noexcept { return bounds.count; }

	__device__ __host__ inline int GetLeftFirst() const noexcept { return bounds.leftFirst; }

	void Subdivide(const AABB * aabbs, BVHNode * bvhTree, unsigned int* primIndices, unsigned int depth, std::atomic_int & poolPtr);

	void SubdivideMT(const AABB * aabbs, BVHNode * bvhTree, unsigned int* primIndices, ctpl::ThreadPool * tPool, std::mutex * threadMutex, std::mutex * partitionMutex, unsigned int* threadCount, unsigned int depth, std::atomic_int & poolPtr);

	bool Partition(const AABB * aabbs, BVHNode * bvhTree, unsigned int* primIndices, std::mutex * partitionMutex, int& left, int& right, std::atomic_int & poolPtr);

	bool Partition(const AABB * aabbs, BVHNode * bvhTree, unsigned int* primIndices, int& left, int& right, std::atomic_int & poolPtr);

	void CalculateBounds(const AABB * aabbs, const unsigned int* primitiveIndices);

	__device__ inline static void traverseBVH(const vec3 & org, const vec3 & dir, float* t, int* hit_idx, const SceneData & scene)
	{
		BVHTraversal todo[32];
		int stackPtr = 0;
		float tNear1, tFar1;
		float tNear2, tFar2;

		const vec3 dirInverse = 1.0f / dir;

		todo[stackPtr].nodeIdx = 0;
		while (stackPtr >= 0)
		{
			const auto& node = scene.gpuBvhNodes[todo[stackPtr].nodeIdx];
			stackPtr--;

			if (node.GetCount() > -1)
			{
				for (int i = 0; i < node.GetCount(); i++)
				{
					const int primIdx = scene.gpuPrimIndices[node.GetLeftFirst() + i];
					const auto idx = scene.indices[primIdx];
					if (triangle::intersect(org, dir, t, scene.vertices[idx.x], scene.vertices[idx.y], scene.vertices[idx.z], scene.triangleEpsilon))
						* hit_idx = primIdx;
				}
			}
			else
			{
				bool hitLeft = scene.gpuBvhNodes[node.GetLeftFirst()].Intersect(org, dirInverse, &tNear1, &tFar1);
				bool hitRight = scene.gpuBvhNodes[node.GetLeftFirst() + 1].Intersect(org, dirInverse, &tNear2, &tFar2);

				if (hitLeft && hitRight)
				{
					if (tNear1 < tNear2)
					{
						stackPtr++;
						todo[stackPtr] = { node.GetLeftFirst(), tNear1 };
						stackPtr++;
						todo[stackPtr] = { node.GetLeftFirst() + 1, tNear2 };
					}
					else
					{
						stackPtr++;
						todo[stackPtr] = { node.GetLeftFirst() + 1, tNear2 };
						stackPtr++;
						todo[stackPtr] = { node.GetLeftFirst(), tNear1 };
					}
				}
				else if (hitLeft)
				{
					stackPtr++;
					todo[stackPtr] = { node.GetLeftFirst(), tNear1 };
				}
				else if (hitRight)
				{
					stackPtr++;
					todo[stackPtr] = { node.GetLeftFirst() + 1, tNear2 };
				}
			}
		}
	}

	__device__ inline static bool traverseBVHShadow(const vec3 & org, const vec3 & dir, float maxDist, const SceneData & scene)
	{
		BVHTraversal todo[32];
		int stackPtr = 0;
		float tNear1, tFar1;
		float tNear2, tFar2;

		const vec3 dirInverse = 1.0f / dir;

		todo[stackPtr].nodeIdx = 0;
		while (stackPtr >= 0)
		{
			const auto& node = scene.gpuBvhNodes[todo[stackPtr].nodeIdx];
			stackPtr--;

			if (node.GetCount() > -1)
			{
				for (int i = 0; i < node.GetCount(); i++)
				{
					const int primIdx = scene.gpuPrimIndices[node.GetLeftFirst() + i];
					const auto idx = scene.indices[primIdx];
					if (triangle::intersect(org, dir, &maxDist, scene.vertices[idx.x], scene.vertices[idx.y], scene.vertices[idx.z], scene.triangleEpsilon))
						return false;
				}
			}
			else
			{
				bool hitLeft = scene.gpuBvhNodes[node.GetLeftFirst()].Intersect(org, dirInverse, &tNear1, &tFar1);
				bool hitRight = scene.gpuBvhNodes[node.GetLeftFirst() + 1].Intersect(org, dirInverse, &tNear2, &tFar2);

				if (hitLeft && hitRight)
				{
					if (tNear1 < tNear2)
					{
						stackPtr++;
						todo[stackPtr] = { node.GetLeftFirst(), tNear1 };
						stackPtr++;
						todo[stackPtr] = { node.GetLeftFirst() + 1, tNear2 };
					}
					else
					{
						stackPtr++;
						todo[stackPtr] = { node.GetLeftFirst() + 1, tNear2 };
						stackPtr++;
						todo[stackPtr] = { node.GetLeftFirst(), tNear1 };
					}
				}
				else if (hitLeft)
				{
					stackPtr++;
					todo[stackPtr] = { node.GetLeftFirst(), tNear1 };
				}
				else if (hitRight)
				{
					stackPtr++;
					todo[stackPtr] = { node.GetLeftFirst() + 1, tNear2 };
				}
			}
		}

		return true;
	}

	__host__ inline static void traverseBVH(const vec3 & org, const vec3 & dir, float* t, int* hit_idx, const std::vector<BVHNode> & nodes, const unsigned int* primIndices, const TriangleList & list)
	{
		BVHTraversal todo[64];
		int stackPtr = 0;
		float tNear1, tFar1;
		float tNear2, tFar2;
		const vec3 dirInverse = 1.0f / dir;

		todo[stackPtr].nodeIdx = 0;
		while (stackPtr >= 0)
		{
			const auto& node = nodes[todo[stackPtr].nodeIdx];
			stackPtr--;

			if (node.GetCount() > -1)
			{
				for (int i = 0; i < node.GetCount(); i++)
				{
					const int primIdx = primIndices[node.GetLeftFirst() + i];
					const auto& idx = list.m_Indices[primIdx];
					if (triangle::intersect(org, dir, t, list.m_Vertices[idx.x], list.m_Vertices[idx.y], list.m_Vertices[idx.z], EPSILON_T))
						* hit_idx = primIdx;
				}
			}
			else
			{
				bool hitLeft = nodes[node.GetLeftFirst()].Intersect(org, dirInverse, &tNear1, &tFar1);
				bool hitRight = nodes[node.GetLeftFirst() + 1].Intersect(org, dirInverse, &tNear2, &tFar2);

				if (hitLeft && hitRight)
				{
					if (tNear1 < tNear2)
					{
						stackPtr++, todo[stackPtr] = { node.GetLeftFirst(), tNear1 };
						stackPtr++, todo[stackPtr] = { node.GetLeftFirst() + 1, tNear2 };
					}
					else
					{
						stackPtr++, todo[stackPtr] = { node.GetLeftFirst() + 1, tNear2 };
						stackPtr++, todo[stackPtr] = { node.GetLeftFirst(), tNear1 };
					}
				}
				else if (hitLeft)
					stackPtr++, todo[stackPtr] = { node.GetLeftFirst(), tNear1 };
				else if (hitRight)
					stackPtr++, todo[stackPtr] = { node.GetLeftFirst() + 1, tNear2 };
			}
		}
	}
};