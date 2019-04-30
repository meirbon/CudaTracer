#pragma once

#include <GL/glew.h>

#include <tuple>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <Tracer/CUDA/Kernel.cuh>
#include <Tracer/CUDA/CudaRenderer.h>
#include <Tracer/Utils/Window.h>
#include <Tracer/Utils/ctpl.h>
#include <Tracer/Utils/Timer.h>

#include <Tracer/Core/TriangleList.h>
#include <Tracer/Core/Camera.h>
#include <Tracer/Core/Material.cuh>
#include <Tracer/Core/Microfacet.cuh>

#include <Tracer/BVH/BVHTree.h>
#include <Tracer/BVH/MBVHTree.h>

class Application
{
public:
	Application(utils::Window* window, TriangleList* tList);
	~Application() = default;

	void loadSkybox(const std::string& path);

	void run();

private:
	void init();
	void allocateBuffers();
	void allocateTriangles();
	void allocateTextures();
	void free();
	void updateMaterials(int matIdx, Material* mat, microfacet::Microfacet* mfMat);
	void drawUI();

	CudaRenderer m_RenderView;

	float m_MovementSpeed = 1.0f;
	float m_Elapsed = 0.0f;

	std::vector<bool> m_Keys;
	glm::dvec2 m_MousePos = glm::dvec2(0.0, 0.0);

	ctpl::ThreadPool* m_ThreadPool;
	MBVHTree* m_MBVHTree = nullptr;
	BVHTree* m_BVHTree = nullptr;
	utils::Window* m_Window;
	utils::Timer m_Timer;

	Params m_Params;
	core::Camera m_Camera;
	TriangleList* m_TriangleList;

	int m_RayBufferSize;
	int m_MaxBufferSize;

	int m_SelectedMatIdx;
	std::pair<Material*, microfacet::Microfacet*> m_SelectedMat;

	void getMaterialAtPixel(const MBVHNode* nodes, const unsigned int* primIndices, TriangleList& tList, Camera& camera, int x, int y);
};
