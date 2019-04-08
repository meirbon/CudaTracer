#include "MBVHNode.cuh"
#include "MBVHTree.h"
#include "../Core/SceneData.cuh"

void MBVHNode::SetBounds(unsigned int nodeIdx, const vec3 &min, const vec3 &max)
{
	this->bminx[nodeIdx] = min.x;
	this->bminy[nodeIdx] = min.y;
	this->bminz[nodeIdx] = min.z;

	this->bmaxx[nodeIdx] = max.x;
	this->bmaxy[nodeIdx] = max.y;
	this->bmaxz[nodeIdx] = max.z;
}

void MBVHNode::SetBounds(unsigned int nodeIdx, const AABB &bounds)
{
	this->bminx[nodeIdx] = bounds.xMin;
	this->bminy[nodeIdx] = bounds.yMin;
	this->bminz[nodeIdx] = bounds.zMin;

	this->bmaxx[nodeIdx] = bounds.xMax;
	this->bmaxy[nodeIdx] = bounds.yMax;
	this->bmaxz[nodeIdx] = bounds.zMax;
}

void MBVHNode::MergeNodes(const BVHNode &node, const BVHNode *bvhPool, MBVHTree *bvhTree)
{
	int numChildren;
	GetBVHNodeInfo(node, bvhPool, numChildren);

	for (int idx = 0; idx < numChildren; idx++)
	{
		if (this->count[idx] == -1)
		{ // not a leaf
			const BVHNode &curNode = bvhPool[this->child[idx]];
			if (curNode.IsLeaf())
			{
				this->count[idx] = curNode.GetCount();
				this->child[idx] = curNode.GetLeftFirst();
				this->SetBounds(idx, curNode.bounds);
			}
			else
			{
				const uint newIdx = bvhTree->m_FinalPtr++;
				MBVHNode &newNode = bvhTree->m_Tree[newIdx];
				this->child[idx] = newIdx; // replace BVHNode idx with MBVHNode idx
				this->count[idx] = -1;
				this->SetBounds(idx, curNode.bounds);
				newNode.MergeNodes(curNode, bvhPool, bvhTree);
			}
		}
	}

	// invalidate any remaining children
	for (int idx = numChildren; idx < 4; idx++)
	{
		this->SetBounds(idx, vec3(1e34f), vec3(-1e34f));
		this->count[idx] = 0;
	}
}

void MBVHNode::MergeNodesMT(const BVHNode &node, const BVHNode *bvhPool, MBVHTree *bvhTree, bool thread)
{
	int numChildren;
	GetBVHNodeInfo(node, bvhPool, numChildren);

	int threadCount = 0;
	std::vector<std::future<void>> threads{};

	// invalidate any remaining children
	for (int idx = numChildren; idx < 4; idx++)
	{
		this->SetBounds(idx, vec3(1e34f), vec3(-1e34f));
		this->count[idx] = 0;
	}

	for (int idx = 0; idx < numChildren; idx++)
	{
		if (this->count[idx] == -1)
		{ // not a leaf
			const BVHNode *curNode = &bvhPool[this->child[idx]];

			if (curNode->IsLeaf())
			{
				this->count[idx] = curNode->GetCount();
				this->child[idx] = curNode->GetLeftFirst();
				this->SetBounds(idx, curNode->bounds);
				continue;
			}

			bvhTree->m_PoolPtrMutex.lock();
			const auto newIdx = bvhTree->m_FinalPtr++;
			bvhTree->m_PoolPtrMutex.unlock();

			MBVHNode *newNode = &bvhTree->m_Tree[newIdx];
			this->child[idx] = newIdx; // replace BVHNode idx with MBVHNode idx
			this->count[idx] = -1;
			this->SetBounds(idx, curNode->bounds);

			if (bvhTree->m_ThreadLimitReached || !thread)
			{
				newNode->MergeNodesMT(*curNode, bvhPool, bvhTree, !thread);
			}
			else
			{
				bvhTree->m_ThreadMutex.lock();
				bvhTree->m_BuildingThreads++;
				if (bvhTree->m_BuildingThreads > ctpl::nr_of_cores)
					bvhTree->m_ThreadLimitReached = true;
				bvhTree->m_ThreadMutex.unlock();

				threadCount++;
				threads.push_back(bvhTree->m_OriginalTree->m_ThreadPool->push(
					[newNode, curNode, bvhPool, bvhTree](int) { newNode->MergeNodesMT(*curNode, bvhPool, bvhTree); }));
			}
		}
	}

	for (int i = 0; i < threadCount; i++)
	{
		threads[i].get();
	}
}

void MBVHNode::MergeNodes(const BVHNode &node, const std::vector<BVHNode> &bvhPool, MBVHTree *bvhTree)
{
	int numChildren;
	GetBVHNodeInfo(node, bvhPool.data(), numChildren);

	for (int idx = 0; idx < numChildren; idx++)
	{
		if (this->count[idx] == -1)
		{ // not a leaf
			const BVHNode &curNode = bvhPool[this->child[idx]];
			if (curNode.IsLeaf())
			{
				this->count[idx] = curNode.GetCount();
				this->child[idx] = curNode.GetLeftFirst();
				this->SetBounds(idx, curNode.bounds);
			}
			else
			{
				const uint newIdx = bvhTree->m_FinalPtr++;
				MBVHNode &newNode = bvhTree->m_Tree[newIdx];
				this->child[idx] = newIdx; // replace BVHNode idx with MBVHNode idx
				this->count[idx] = -1;
				this->SetBounds(idx, curNode.bounds);
				newNode.MergeNodes(curNode, bvhPool, bvhTree);
			}
		}
	}

	// invalidate any remaining children
	for (int idx = numChildren; idx < 4; idx++)
	{
		this->SetBounds(idx, vec3(1e34f), vec3(-1e34f));
		this->count[idx] = 0;
	}
}

void MBVHNode::MergeNodesMT(const BVHNode &node, const std::vector<BVHNode> &bvhPool, MBVHTree *bvhTree,
	bool thread)
{
	int numChildren;
	GetBVHNodeInfo(node, bvhPool.data(), numChildren);

	int threadCount = 0;
	std::vector<std::future<void>> threads{};

	// Invalidate any remaining children
	for (int idx = numChildren; idx < 4; idx++)
	{
		this->SetBounds(idx, vec3(1e34f), vec3(-1e34f));
		this->count[idx] = 0;
	}

	for (int idx = 0; idx < numChildren; idx++)
	{
		if (this->count[idx] == -1)
		{ // not a leaf
			const BVHNode *curNode = &bvhPool[this->child[idx]];

			if (curNode->IsLeaf())
			{
				this->count[idx] = curNode->GetCount();
				this->child[idx] = curNode->GetLeftFirst();
				this->SetBounds(idx, curNode->bounds);
				continue;
			}

			bvhTree->m_PoolPtrMutex.lock();
			const auto newIdx = bvhTree->m_FinalPtr++;
			bvhTree->m_PoolPtrMutex.unlock();

			MBVHNode *newNode = &bvhTree->m_Tree[newIdx];
			this->child[idx] = newIdx; // replace BVHNode idx with MBVHNode idx
			this->count[idx] = -1;
			this->SetBounds(idx, curNode->bounds);

			if (bvhTree->m_ThreadLimitReached || !thread)
			{
				newNode->MergeNodesMT(*curNode, bvhPool, bvhTree, !thread);
			}
			else
			{
				bvhTree->m_ThreadMutex.lock();
				bvhTree->m_BuildingThreads++;
				if (bvhTree->m_BuildingThreads > ctpl::nr_of_cores)
					bvhTree->m_ThreadLimitReached = true;
				bvhTree->m_ThreadMutex.unlock();

				threadCount++;
				threads.push_back(bvhTree->m_OriginalTree->m_ThreadPool->push(
					[newNode, curNode, bvhPool, bvhTree](int) { newNode->MergeNodesMT(*curNode, bvhPool, bvhTree); }));
			}
		}
	}

	for (int i = 0; i < threadCount; i++)
	{
		threads[i].get();
	}
}

void MBVHNode::GetBVHNodeInfo(const BVHNode &node, const BVHNode *pool, int &numChildren)
{
	// Starting values
	child[0] = child[1] = child[2] = child[3] = -1;
	count[0] = count[1] = count[2] = count[3] = -1;
	numChildren = 0;

	if (node.IsLeaf())
	{
		std::cout << "This node shouldn't be a leaf." << "MBVHNode" << std::endl;
		return;
	}

	const BVHNode &leftNode = pool[node.GetLeftFirst()];
	const BVHNode &rightNode = pool[node.GetLeftFirst() + 1];

	if (leftNode.IsLeaf())
	{
		// node only has a single child
		const int idx = numChildren++;
		SetBounds(idx, leftNode.bounds);
		child[idx] = leftNode.GetLeftFirst();
		count[idx] = leftNode.GetCount();
	}
	else
	{
		// Node has 2 children
		const int idx1 = numChildren++;
		const int idx2 = numChildren++;
		child[idx1] = leftNode.GetLeftFirst();
		child[idx2] = leftNode.GetLeftFirst() + 1;
	}

	if (rightNode.IsLeaf())
	{
		// Node only has a single child
		const int idx = numChildren++;
		SetBounds(idx, rightNode.bounds);
		child[idx] = rightNode.GetLeftFirst();
		count[idx] = rightNode.GetCount();
	}
	else
	{
		// Node has 2 children
		const int idx1 = numChildren++;
		const int idx2 = numChildren++;
		SetBounds(idx1, pool[rightNode.GetLeftFirst()].bounds);
		SetBounds(idx2, pool[rightNode.GetLeftFirst() + 1].bounds);
		child[idx1] = rightNode.GetLeftFirst();
		child[idx2] = rightNode.GetLeftFirst() + 1;
	}
}

void MBVHNode::SortResults(const float *tmin, int &a, int &b, int &c, int &d) const
{
	if (tmin[a] > tmin[b])
		std::swap(a, b);
	if (tmin[c] > tmin[d])
		std::swap(c, d);
	if (tmin[a] > tmin[c])
		std::swap(a, c);
	if (tmin[b] > tmin[d])
		std::swap(b, d);
	if (tmin[b] > tmin[c])
		std::swap(b, c);
}