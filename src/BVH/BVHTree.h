#pragma once

#include <iostream>
#include <vector>
#include <mutex>

#include "Core/TriangleList.h"
#include "Utils/ctpl.h"
#include "BVH/BVHNode.cuh"
#include "BVH/AABB.h"

class BVHTree
{
public:
	BVHTree(TriangleList* objectList, ctpl::ThreadPool* pool = nullptr);

	void constructBVH();
	void buildBVH(std::vector<AABB>* aabbs);
	void reset();

public:
	std::vector<BVHNode> m_BVHPool;
	std::vector<unsigned int> m_PrimitiveIndices;
	size_t m_PrimitiveCount = 0;
	TriangleList* m_ObjectList;

	bool CanUseBVH = false;
	std::atomic_int m_PoolPtr;

	ctpl::ThreadPool* m_ThreadPool = nullptr;
	std::mutex m_PoolPtrMutex{};
	std::mutex m_ThreadMutex{};
	unsigned int m_BuildingThreads = 0;
	bool m_ThreadLimitReached = false;
};