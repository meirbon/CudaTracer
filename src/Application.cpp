#include <Tracer/Application.h>

#include <iostream>

#include <Tracer/CUDA/CudaAssert.h>
#include "ImGui/imgui.h"

using namespace core;
using namespace glm;
using namespace microfacet;

Application::Application(utils::Window* window, TriangleList* tList)
	: m_RenderView(window->getWidth(), window->getHeight())
	, m_ThreadPool(new ctpl::ThreadPool(ctpl::nr_of_cores))
	, m_Window(window)
	, m_Params({ window->getWidth(), window->getHeight() })
	, m_TriangleList(tList)
{
	m_MaxBufferSize = m_RayBufferSize = m_Params.width * m_Params.height * 2;
	m_Camera = Camera(m_Params.width, m_Params.height, 70.0f);
	m_Keys.resize(1024);
	m_Params.camera = &m_Camera;
	m_Params.reset();

	m_Window->setEventCallback([this](utils::Window::Event event)
		{
			switch (event.type) {
			case(utils::Window::CLOSED):
				m_Window->close();
				break;
			case(utils::Window::MOUSE):
				if (event.state == utils::Window::MOUSE_MOVE)
				{
					m_MousePos = { event.realX, event.realY };
					if (m_Keys[GLFW_MOUSE_BUTTON_RIGHT])
					{
						if (m_Camera.processMouse(event.x, -event.y))
							m_Params.reset();
					}
				}
				else if (event.state == utils::Window::MOUSE_SCROLL)
				{
					if (m_Camera.changeFov(event.y))
						m_Params.reset();
				}

				if (event.key < 0) break;
				if (event.state == utils::Window::KEY_PRESSED)
					m_Keys[event.key] = true;
				else if (event.state == utils::Window::KEY_RELEASED)
					m_Keys[event.key] = false;
				break;
			case(utils::Window::KEY): {
				if (event.key < 0) break;
				if (event.state == utils::Window::KEY_PRESSED)
					m_Keys[event.key] = true;
				else if (event.state == utils::Window::KEY_RELEASED)
					m_Keys[event.key] = false;
			}
			}
		}
	);

	m_Window->setResizeCallback([this](int width, int height)
		{
			m_Camera.setDimensions(width, height);
			m_Params.width = width, m_Params.height = height;

			m_RayBufferSize = width * height;
			m_MaxBufferSize = width * height * 2;

			allocateBuffers();
			m_RenderView.setDimensions(width, height);
		}
	);
}

void Application::loadSkybox(const std::string & path)
{
	cuda(DeviceSynchronize());
	if (m_Params.gpuScene.skyboxTexture >= 0) // Overwrite previous sky box if it exists
		m_Params.gpuScene.skyboxTexture = m_TriangleList->overwriteTexture(m_Params.gpuScene.skyboxTexture, path);
	else
		m_Params.gpuScene.skyboxTexture = m_TriangleList->loadTexture(path);
	m_Params.gpuScene.skyboxEnabled = m_Params.gpuScene.skyboxTexture >= 0;

	if (m_Params.gpuScene.skyboxEnabled)
		allocateTextures();
}

void Application::run()
{
	init();
	std::cout << "Application init done." << std::endl;

	const vec4 black = vec4(0.0f);
	std::pair<Material*, Microfacet*> selectedMaterial = std::make_pair(nullptr, nullptr);

	while (!m_Window->shouldClose())
	{
		m_Elapsed = m_Timer.elapsed();
		m_Timer.reset();
		m_Window->pollEvents();

		if (m_Keys[GLFW_KEY_ESCAPE] || m_Keys[GLFW_KEY_Q])
			m_Window->close();
		if (m_Keys[GLFW_KEY_LEFT_ALT] && m_Keys[GLFW_KEY_ENTER])
			m_Window->switchFullscreen();
		if (m_Keys[GLFW_KEY_R])
			m_Params.reset();

		auto uiDrawn = m_ThreadPool->push([this](int) { drawUI(); });

		if (m_Params.samples < m_MaxFrameCount || m_MaxFrameCount == 0)
			launchKernels(m_RenderView.getCudaArray(), m_Params, m_RayBufferSize);

		m_RenderView.draw();
		uiDrawn.get();

		m_Window->present();
		if (m_Camera.handleKeys(m_Keys, m_Elapsed))
			m_Params.reset();
	}

	free();
}

void Application::init()
{
	m_BVHTree = new BVHTree(m_TriangleList, m_ThreadPool);
	m_BVHTree->constructBVH();
	m_MBVHTree = new MBVHTree(m_BVHTree);
	m_MBVHTree->constructBVH();

	allocateBuffers();
	allocateTriangles();
}

void Application::allocateBuffers()
{
	cuda(DeviceSynchronize());

	cudaFree(m_Params.gpuScene.currentFrame);
	cudaFree(m_Params.gpuRays);
	cudaFree(m_Params.gpuNextRays);
	cudaFree(m_Params.gpuShadowRays);

	cuda(Malloc(&m_Params.gpuRays, m_MaxBufferSize * sizeof(Ray)));
	cuda(Malloc(&m_Params.gpuNextRays, m_MaxBufferSize * sizeof(Ray)));
	cuda(Malloc(&m_Params.gpuShadowRays, m_MaxBufferSize * sizeof(ShadowRay)));
	cuda(Malloc(&m_Params.gpuScene.currentFrame, m_Params.width * m_Params.height * sizeof(vec4)));
}

void Application::allocateTriangles()
{
	cuda(Malloc(&m_Params.gpuScene.gpuMbvhNodes, m_MBVHTree->m_FinalPtr * sizeof(MBVHNode)));
	cuda(MemcpyAsync(m_Params.gpuScene.gpuMbvhNodes, m_MBVHTree->m_Tree.data(), m_MBVHTree->m_FinalPtr * sizeof(MBVHNode), cudaMemcpyHostToDevice));

	cuda(Malloc(&m_Params.gpuScene.gpuBvhNodes, m_BVHTree->m_BVHPool.size() * sizeof(BVHNode)));
	cuda(MemcpyAsync(m_Params.gpuScene.gpuBvhNodes, m_BVHTree->m_BVHPool.data(), m_BVHTree->m_BVHPool.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));

	cuda(Malloc(&m_Params.gpuScene.gpuPrimIndices, m_BVHTree->m_PrimitiveIndices.size() * sizeof(int)));
	cuda(MemcpyAsync(m_Params.gpuScene.gpuPrimIndices, m_BVHTree->m_PrimitiveIndices.data(), m_BVHTree->m_PrimitiveIndices.size() * sizeof(int), cudaMemcpyHostToDevice));

	cuda(Malloc(&m_Params.gpuScene.gpuMaterials, m_TriangleList->m_Materials.size() * sizeof(Material)));
	cuda(MemcpyAsync(m_Params.gpuScene.gpuMaterials, m_TriangleList->m_Materials.data(), m_TriangleList->m_Materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

	cuda(Malloc(&m_Params.gpuScene.vertices, m_TriangleList->m_Vertices.size() * sizeof(vec3)));
	cuda(MemcpyAsync(m_Params.gpuScene.vertices, m_TriangleList->m_Vertices.data(), m_TriangleList->m_Vertices.size() * sizeof(vec3), cudaMemcpyHostToDevice));

	cuda(Malloc(&m_Params.gpuScene.normals, m_TriangleList->m_Normals.size() * sizeof(vec3)));
	cuda(MemcpyAsync(m_Params.gpuScene.normals, m_TriangleList->m_Normals.data(), m_TriangleList->m_Normals.size() * sizeof(vec3), cudaMemcpyHostToDevice));

	cuda(Malloc(&m_Params.gpuScene.centerNormals, m_TriangleList->m_CenterNormals.size() * sizeof(vec3)));
	cuda(MemcpyAsync(m_Params.gpuScene.centerNormals, m_TriangleList->m_CenterNormals.data(), m_TriangleList->m_CenterNormals.size() * sizeof(vec3), cudaMemcpyHostToDevice));

	cuda(Malloc(&m_Params.gpuScene.texCoords, m_TriangleList->m_TexCoords.size() * sizeof(vec2)));
	cuda(MemcpyAsync(m_Params.gpuScene.texCoords, m_TriangleList->m_TexCoords.data(), m_TriangleList->m_TexCoords.size() * sizeof(vec2), cudaMemcpyHostToDevice));

	cuda(Malloc(&m_Params.gpuScene.indices, m_TriangleList->m_Indices.size() * sizeof(uvec3)));
	cuda(MemcpyAsync(m_Params.gpuScene.indices, m_TriangleList->m_Indices.data(), m_TriangleList->m_Indices.size() * sizeof(uvec3), cudaMemcpyHostToDevice));

	cuda(Malloc(&m_Params.gpuScene.lightIndices, m_TriangleList->m_LightIndices.size() * sizeof(unsigned int)));
	cuda(MemcpyAsync(m_Params.gpuScene.lightIndices, m_TriangleList->m_LightIndices.data(), m_TriangleList->m_LightIndices.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

	cuda(Malloc(&m_Params.gpuScene.gpuMatIdxs, m_TriangleList->m_MaterialIdxs.size() * sizeof(unsigned int)));
	cuda(MemcpyAsync(m_Params.gpuScene.gpuMatIdxs, m_TriangleList->m_MaterialIdxs.data(), m_TriangleList->m_MaterialIdxs.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

	cuda(Malloc(&m_Params.gpuScene.microfacets, m_TriangleList->m_Microfacets.size() * sizeof(microfacet::Microfacet)));
	cuda(MemcpyAsync(m_Params.gpuScene.microfacets, m_TriangleList->m_Microfacets.data(), m_TriangleList->m_Microfacets.size() * sizeof(microfacet::Microfacet), cudaMemcpyHostToDevice));

	m_Params.gpuScene.lightCount = m_TriangleList->m_LightIndices.size();
	cuda(DeviceSynchronize());
}

void Application::allocateTextures()
{
	cuda(DeviceSynchronize());
	cuda(Free(m_Params.gpuScene.gpuTexBuffer));
	cuda(Free(m_Params.gpuScene.gpuTexDims));
	cuda(Free(m_Params.gpuScene.gpuTexOffsets));

	const auto textureBuffer = m_TriangleList->createTextureBuffer();

	cuda(Malloc(&m_Params.gpuScene.gpuTexBuffer, textureBuffer.textureColors.size() * sizeof(vec4)));
	cuda(MemcpyAsync(m_Params.gpuScene.gpuTexBuffer, textureBuffer.textureColors.data(), textureBuffer.textureColors.size() * sizeof(vec4), cudaMemcpyHostToDevice));

	cuda(Malloc(&m_Params.gpuScene.gpuTexDims, textureBuffer.textureDims.size() * sizeof(uint)));
	cuda(MemcpyAsync(m_Params.gpuScene.gpuTexDims, textureBuffer.textureDims.data(), textureBuffer.textureDims.size() * sizeof(uint), cudaMemcpyHostToDevice));

	cuda(Malloc(&m_Params.gpuScene.gpuTexOffsets, textureBuffer.textureOffsets.size() * sizeof(uint)));
	cuda(MemcpyAsync(m_Params.gpuScene.gpuTexOffsets, textureBuffer.textureOffsets.data(), textureBuffer.textureOffsets.size() * sizeof(uint), cudaMemcpyHostToDevice));
}

void Application::free()
{
	m_Params.deallocate();
	delete m_MBVHTree;
	delete m_BVHTree;
	delete m_ThreadPool;
}

void Application::updateMaterials(int matIdx, Material * mat, microfacet::Microfacet * mfMat)
{
	cuda(MemcpyAsync(&m_Params.gpuScene.microfacets[matIdx], mfMat, sizeof(microfacet::Microfacet), cudaMemcpyHostToDevice));
	cuda(MemcpyAsync(&m_Params.gpuScene.gpuMaterials[matIdx], mat, sizeof(Material), cudaMemcpyHostToDevice));
}

void Application::drawUI()
{
	ImGui::Begin("Settings");
	if (ImGui::Checkbox("Sky box", &m_Params.gpuScene.skyboxEnabled))
		m_Params.reset();
	if (ImGui::Checkbox("Reference", &m_Params.reference))
		m_Params.reset();
	if (ImGui::Checkbox("Indirect", &m_Params.gpuScene.indirect))
		m_Params.reset();
	if (ImGui::Checkbox("Shadow", &m_Params.gpuScene.shadow))
		m_Params.reset();
	if (ImGui::DragInt("Max frames", &m_MaxFrameCount, 1))
		m_Params.reset();

	ImGui::DragFloat("Speed", &m_MovementSpeed, 0.2f, 0.2f, 100.f);
	vec3 pos = m_Camera.getPosition();
	ImGui::DragFloat3("Position", glm::value_ptr(pos));
	ImGui::DragInt("RayBufferSize", &m_RayBufferSize, 16384.f, 1024, m_MaxBufferSize);
	m_RayBufferSize = min(m_RayBufferSize, m_MaxBufferSize);
	ImGui::DragFloat("N Epsilon", &m_Params.gpuScene.normalEpsilon, 0.00001f, 1e-8f, 1e-2f, "%.8f");
	ImGui::DragFloat("ShadowDist Epsilon", &m_Params.gpuScene.distEpsilon, 0.00001f, 1e-8f, 1e-2f, "%.8f");
	ImGui::DragFloat("Triangle Epsilon", &m_Params.gpuScene.triangleEpsilon, 0.00001f, 1e-8f, 1e-2f, "%.8f");

	if (m_Keys[GLFW_KEY_T])
		getMaterialAtPixel(m_MBVHTree->m_Tree.data(), m_MBVHTree->m_PrimitiveIndices.data(), *m_TriangleList, m_Camera, float(m_MousePos.x), float(m_MousePos.y));

	auto* mat = std::get<0>(m_SelectedMat);
	auto* mfMat = std::get<1>(m_SelectedMat);

	if (mat != nullptr && mfMat != nullptr)
	{
		ImGui::BeginChild("Material");

		bool updateMat = false;
		const bool aV = mfMat->sampleVisibility;
		const float aX = mfMat->alphaX;
		const float aY = mfMat->alphaY;
		const auto matTypes = Material::getTypes();
		static bool linkValues = false;

		ImGui::Text("MatType: %i", mat->type);
		ImGui::Text("Type: %s", Material::getTypeName(mat->type));

		int matType = mat->type;
		if (ImGui::ListBox("Type", &matType, matTypes.data(), int(matTypes.size()), 3))
		{
			updateMat = true;
			const std::string newType = matTypes[matType];
			mat->changeType(newType);
		}

		updateMat |= ImGui::DragFloat("RefractIdx", &mat->refractIdx, 0.1f, 1.0f, 5.0f);
		updateMat |= ImGui::DragFloat3("Color", glm::value_ptr(mat->albedo), 0.1f, 0.0f, 1000.0f);
		updateMat |= ImGui::Checkbox("Link values", &linkValues);
		updateMat |= ImGui::Checkbox("Sample Visibility", &mfMat->sampleVisibility);
		updateMat |= ImGui::DragFloat2("Roughness", glm::value_ptr(mfMat->alpha), 0.01f, 1e-6f, 1.0f);

		if (mat->type == Fresnel || mat->type >= FresnelBeckmann)
			updateMat |= ImGui::DragFloat3("Absorption", glm::value_ptr(mat->absorption), 0.1f, 0.0f, 100.0f);

		if (updateMat)
		{
			if (linkValues) mfMat->alphaY = mfMat->alphaX;
			updateMaterials(m_SelectedMatIdx, mat, mfMat);
			m_Params.reset();
		}

		ImGui::EndChild();
	}

	ImGui::End();
	cuda(DeviceSynchronize());
}

void Application::getMaterialAtPixel(const MBVHNode * nodes, const unsigned int* primIndices, TriangleList & tList, Camera & camera, int x, int y)
{
	Ray ray = camera.generateRay(float(x), float(y));
	MBVHNode::traverseMBVH(ray.origin, ray.direction, &ray.t, &ray.hit_idx, nodes, primIndices, tList);

	if (ray.valid())
	{
		m_SelectedMatIdx = tList.m_MaterialIdxs[ray.hit_idx];
		m_SelectedMat = std::make_pair(&tList.m_Materials[m_SelectedMatIdx], &tList.m_Microfacets[m_SelectedMatIdx]);
	}
	else
	{
		m_SelectedMatIdx = -1;
		m_SelectedMat = std::make_pair(nullptr, nullptr);
	}
}
