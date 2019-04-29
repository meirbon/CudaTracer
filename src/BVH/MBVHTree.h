#pragma once

#include <mutex>

#include "BVH/MBVHNode.cuh"
#include "BVH/BVHTree.h"
#include "Utils/ctpl.h"

class AABB;
class MBVHNode;

class MBVHTree
{
public:
	friend class MBVHNode;

	explicit MBVHTree(BVHTree *orgTree);

	BVHTree *m_OriginalTree;
	std::vector<MBVHNode> m_Tree;
	std::vector<unsigned int> m_PrimitiveIndices{};

	bool m_CanUseBVH = false;
	unsigned int m_FinalPtr = 0;

	void constructBVH();
private:
	std::mutex m_PoolPtrMutex{};
	std::mutex m_ThreadMutex{};
	unsigned int m_BuildingThreads = 0;
	bool m_ThreadLimitReached = false;
};
