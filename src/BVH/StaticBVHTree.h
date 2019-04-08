#pragma once

#include <iostream>
#include <vector>

#include "../TriangleList.h"
#include "BVHNode.cuh"
#include "../Utils/ctpl.h"
#include "AABB.h"

class StaticBVHTree
{
public:
	StaticBVHTree(TriangleList *objectList, BVHType type = SAH, ctpl::ThreadPool *pool = nullptr);

	void ConstructBVH();
	void BuildBVH(std::vector<AABB> *aabbs);
	void Reset();

public:
	std::vector<BVHNode> m_BVHPool;
	std::vector<unsigned int> m_PrimitiveIndices;
	size_t m_PrimitiveCount = 0;
	TriangleList *m_ObjectList;

	bool CanUseBVH = false;
	std::atomic_int m_PoolPtr = 0;

	BVHType m_Type = SAH;
	ctpl::ThreadPool *m_ThreadPool = nullptr;
	std::mutex m_PoolPtrMutex{};
	std::mutex m_ThreadMutex{};
	unsigned int m_BuildingThreads = 0;
	bool m_ThreadLimitReached = false;
};