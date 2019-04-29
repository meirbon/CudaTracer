#include "BVH/BVHNode.cuh"
#include "BVH/BVHTree.h"

#define MAX_PRIMS 4
#define MAX_DEPTH 64
#define BINS 11

using namespace glm;

BVHNode::BVHNode()
{
	SetLeftFirst(-1);
	SetCount(-1);
}

BVHNode::BVHNode(int leftFirst, int count, AABB bounds)
	: bounds(bounds)
{
	SetLeftFirst(leftFirst);
	SetCount(-1);
}

void BVHNode::CalculateBounds(const AABB* aabbs, const unsigned int* primitiveIndices)
{
	AABB newBounds = { vec3(1e34f), vec3(-1e34f) };
	for (int idx = 0; idx < bounds.count; idx++) {
		newBounds.Grow(aabbs[primitiveIndices[bounds.leftFirst + idx]]);
	}

	bounds.xMin = newBounds.xMin;
	bounds.yMin = newBounds.yMin;
	bounds.zMin = newBounds.zMin;
	bounds.xMax = newBounds.xMax;
	bounds.yMax = newBounds.yMax;
	bounds.zMax = newBounds.zMax;
}

void BVHNode::Subdivide(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, unsigned int depth, std::atomic_int &poolPtr)
{
	depth++;
	if (GetCount() < MAX_PRIMS || depth >= MAX_DEPTH)
		return; // this is a leaf node

	int left = -1;
	int right = -1;

	if (!Partition(aabbs, bvhTree, primIndices, left, right, poolPtr)) {
		return;
	}

	this->bounds.leftFirst = left; // set pointer to children
	this->bounds.count = -1; // no primitives since we are no leaf node

	auto& leftNode = bvhTree[left];
	auto& rightNode = bvhTree[right];

	if (leftNode.bounds.count > 0) {
		leftNode.CalculateBounds(aabbs, primIndices);
		leftNode.Subdivide(aabbs, bvhTree, primIndices, depth, poolPtr);
	}

	if (rightNode.bounds.count > 0) {
		rightNode.CalculateBounds(aabbs, primIndices);
		rightNode.Subdivide(aabbs, bvhTree, primIndices, depth, poolPtr);
	}
}

void BVHNode::SubdivideMT(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, ctpl::ThreadPool *tPool, std::mutex *threadMutex, std::mutex *partitionMutex, unsigned int *threadCount, unsigned int depth, std::atomic_int &poolPtr)
{
	depth++;
	if (GetCount() < MAX_PRIMS || depth >= MAX_DEPTH)
		return; // this is a leaf node

	int left = -1;
	int right = -1;

	if (!Partition(aabbs, bvhTree, primIndices, partitionMutex, left, right, poolPtr))
		return;

	this->bounds.leftFirst = left; // set pointer to children
	this->bounds.count = -1; // no primitives since we are no leaf node

	auto* leftNode = &bvhTree[left];
	auto* rightNode = &bvhTree[right];

	const auto subLeft = leftNode->GetCount() > 0;
	const auto subRight = rightNode->GetCount() > 0;

	if ((*threadCount) < ctpl::nr_of_cores) {
		threadMutex->lock();

		if (subLeft && subRight)
			(*threadCount)++;

		threadMutex->unlock();

		auto leftThread = tPool->push([aabbs, bvhTree, primIndices, tPool, threadMutex, partitionMutex, threadCount, depth, leftNode, &poolPtr](int) -> void {
			leftNode->CalculateBounds(aabbs, primIndices);
			leftNode->SubdivideMT(aabbs, bvhTree, primIndices, tPool, threadMutex, partitionMutex, threadCount, depth, poolPtr);
		});

		rightNode->CalculateBounds(aabbs, primIndices);
		rightNode->SubdivideMT(aabbs, bvhTree, primIndices, tPool, threadMutex, partitionMutex, threadCount, depth, poolPtr);
		leftThread.get();
	}
	else {
		if (subLeft) {
			leftNode->CalculateBounds(aabbs, primIndices);
			leftNode->SubdivideMT(aabbs, bvhTree, primIndices, tPool, threadMutex, partitionMutex, threadCount, depth, poolPtr);
		}

		if (subRight) {
			rightNode->CalculateBounds(aabbs, primIndices);
			rightNode->SubdivideMT(aabbs, bvhTree, primIndices, tPool, threadMutex, partitionMutex, threadCount, depth, poolPtr);
		}
	}
}

bool BVHNode::Partition(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, std::mutex *partitionMutex, int &left, int &right, std::atomic_int& poolPtr)
{
	const int lFirst = bounds.leftFirst;
	int lCount = 0;
	int rFirst = bounds.leftFirst;
	int rCount = bounds.count;

	float parentNodeCost{}, lowestNodeCost = 1e34f, bestCoord{};
	int bestAxis{};

	parentNodeCost = bounds.Area() * bounds.count;
	const vec3 lengths = this->bounds.Lengths();
	for (int axis = 0; axis < 3; axis++) {
		for (int i = 1; i < BINS; i++) {
			const auto binOffset = float(i) / float(BINS);
			const float splitCoord = this->min[axis] + lengths[axis] * binOffset;
			int leftCount = 0, rightCount = 0;
			AABB leftBox = { vec3(1e34f), vec3(-1e34f) };
			AABB rightBox = { vec3(1e34f), vec3(-1e34f) };

			for (int idx = 0; idx < bounds.count; idx++) {
				const auto& aabb = aabbs[primIndices[lFirst + idx]];
				if (aabb.Centroid()[axis] <= splitCoord) {
					leftBox.Grow(aabb);
					leftCount++;
				}
				else {
					rightBox.Grow(aabb);
					rightCount++;
				}
			}

			const float leftArea = leftBox.Area();
			const float rightArea = rightBox.Area();

			const float splitNodeCost = leftArea * float(leftCount) + rightArea * float(rightCount);
			if (splitNodeCost < lowestNodeCost) {
				lowestNodeCost = splitNodeCost;
				bestCoord = splitCoord;
				bestAxis = axis;
			}
		}
	}

	if (parentNodeCost < lowestNodeCost)
		return false;

	for (int idx = 0; idx < bounds.count; idx++) {
		const auto& aabb = aabbs[primIndices[lFirst + idx]];

		if (aabb.Centroid()[bestAxis] <= bestCoord) // is on left side
		{
			std::swap(primIndices[lFirst + idx], primIndices[lFirst + lCount]);
			lCount++;
			rFirst++;
			rCount--;
		}
	}

	partitionMutex->lock();
	left = poolPtr++;
	right = poolPtr++;
	partitionMutex->unlock();

	bvhTree[left].bounds.leftFirst = lFirst;
	bvhTree[left].bounds.count = lCount;
	bvhTree[right].bounds.leftFirst = rFirst;
	bvhTree[right].bounds.count = rCount;

	return true;
}
bool BVHNode::Partition(const AABB* aabbs, BVHNode* bvhTree, unsigned int* primIndices, int& left, int& right, std::atomic_int& poolPtr)
{
	const int lFirst = bounds.leftFirst;
	int lCount = 0;
	int rFirst = bounds.leftFirst;
	int rCount = bounds.count;

	float parentNodeCost{}, lowestNodeCost = 1e34f, bestCoord{};
	int bestAxis{};

	parentNodeCost = bounds.Area() * bounds.count;
	const vec3 lengths = this->bounds.Lengths();
	for (int i = 1; i < BINS; i++) {
		const auto binOffset = float(i) / float(BINS);
		for (int axis = 0; axis < 3; axis++) {
			const float splitCoord = this->min[axis] + lengths[axis] * binOffset;
			int leftCount = 0, rightCount = 0;
			AABB leftBox = { vec3(1e34f), vec3(-1e34f) };
			AABB rightBox = { vec3(1e34f), vec3(-1e34f) };

			for (int idx = 0; idx < bounds.count; idx++) {
				const auto& aabb = aabbs[primIndices[lFirst + idx]];
				if (aabb.Centroid()[axis] <= splitCoord) {
					leftBox.Grow(aabb);
					leftCount++;
				}
				else {
					rightBox.Grow(aabb);
					rightCount++;
				}
			}

			const float splitNodeCost = leftBox.Area() * leftCount + rightBox.Area() * rightCount;
			if (splitNodeCost < lowestNodeCost) {
				lowestNodeCost = splitNodeCost;
				bestCoord = splitCoord;
				bestAxis = axis;
			}
		}
	}

	if (parentNodeCost < lowestNodeCost)
		return false;

	for (int idx = 0; idx < bounds.count; idx++) {
		const auto& aabb = aabbs[primIndices[lFirst + idx]];

		if (aabb.Centroid()[bestAxis] <= bestCoord) // is on left side
		{
			std::swap(primIndices[lFirst + idx], primIndices[lFirst + lCount]);
			lCount++;
			rFirst++;
			rCount--;
		}
	}

	left = static_cast<int>(poolPtr++);
	right = static_cast<int>(poolPtr++);

	bvhTree[left].bounds.leftFirst = lFirst;
	bvhTree[left].bounds.count = lCount;
	bvhTree[right].bounds.leftFirst = rFirst;
	bvhTree[right].bounds.count = rCount;

	return true;
}