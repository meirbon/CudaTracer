#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext.hpp>

#include "src/Utils/ctpl.h"
#include "src/kernel.cuh"
#include "src/BVH/AABB.h"
#include "src/BVH/StaticBVHTree.h"
#include "src/BVH/MBVHTree.h"
#include "src/Utils/GLFWWindow.h"
#include "src/Core/Camera.h"

#include "src/cuda_assert.h"
#include "src/CudaRenderer.h"
#include "src/TriangleList.h"

#include "src/Utils/Timer.h"

using namespace utils;
using namespace core;

static int samples = 0;
static bool mouseMoved = false;
static float movementSpeed = 1.0f;
static Params params;
static size_t focusedMatIdx;
static std::tuple<Material*, microfacet::Microfacet*> focusedMat = std::make_tuple(nullptr, nullptr);
#define SCRWIDTH 1280
#define SCRHEIGHT 720

static int rayBufferSize = SCRWIDTH * SCRHEIGHT;
static double mouseX, mouseY;

void allocateBuffers();
std::tuple<Material*, microfacet::Microfacet*> getMaterialAtPixel(const MBVHNode* nodes, const unsigned int* primIndices, TriangleList& tList, Camera& camera, int x, int y);

int main(int argc, char* argv[])
{
	params.width = SCRWIDTH;
	params.height = SCRHEIGHT;

	auto window = GLFWWindow("CudaTracer", params.width, params.height, false, false);
	glfwSwapInterval(0);

	int glDeviceId;
	unsigned int glDeviceCount;
	cudaGLGetDevices(&glDeviceCount, &glDeviceId, 1u, cudaGLDeviceListAll);

	struct cudaDeviceProp props;
	cudaGetDeviceProperties(&props, glDeviceId);
	params.smCores = props.multiProcessorCount;

	std::cout << props.name << ", SM Count: " << params.smCores << std::endl;

	auto camera = core::Camera(params.width, params.height, 70.0f);
	auto triangleList = TriangleList();
	auto* tPool = new ctpl::ThreadPool(ctpl::nr_of_cores);

	const std::string sky = "Models/envmaps/pisa.png";
	params.gpuScene.skyboxTexture = triangleList.loadTexture(sky);
	params.gpuScene.skyboxEnabled = params.gpuScene.skyboxTexture >= 0;

	std::string nanosuit = "Models/nanosuit/nanosuit.obj";
	std::string sponza = "Models/sponza/sponza.obj";
	std::string sphere = "Models/sphere.obj";
	std::string dragon = "Models/dragon.obj";
	std::string cbox = "Models/cornellbox/CornellBox-Original.obj";
	std::string countryKitch = "Models/assets/country_kitchen/Country-Kitchen.obj";
	std::string cat = "Models/assets/egyptcat/egyptcat.obj";
	std::string luxball = "Models/assets/luxball/luxball.obj";
	std::string conference = "Models/assets/conference/conference.obj";
	std::string teapot = "Models/teapot.obj";
	std::string waterCbox = "Models/cbox/CornellBox-Sphere.obj";

	const auto mat = triangleList.addMaterial(Material::light(vec3(1)));
	const auto tmat = triangleList.addMaterial(Material::lambertian(vec3(0.2f), 1.0f));

	triangleList.m_Microfacets[tmat].alphaX = 0.001f;
	triangleList.m_Microfacets[tmat].alphaY = 0.001f;
	triangleList.m_Materials[tmat].type = Fresnel;
	triangleList.m_Materials[tmat].refractIdx = 1.1f;

	//triangleList.loadModel(sphere, 1.0f, glm::translate(mat4(1.0f), vec3(8.0f, 3.7f, 0.0f)), mat);
	triangleList.loadModel(sphere, 10.0f, glm::translate(mat4(1.0f), vec3(4.0f, 40.0f, 0.0f)), mat);
	//triangleList.loadModel(sphere, .1f, glm::translate(mat4(1.0f), vec3(0.0f, 10.f, 0.0f)), mat);
	//triangleList.loadModel(sphere, 1.0f, glm::translate(mat4(1.0f), vec3(-4.0f, 3.7f, 0.0f)), mat);
	//triangleList.loadModel(sphere, 1.0f, glm::translate(mat4(1.0f), vec3(-8.0f, 3.7f, 0.0f)), mat);

	//triangleList.loadModel(waterCbox);
	//triangleList.m_Materials[4].type = Fresnel;
	//triangleList.m_Materials[4].refractIdx = 1.5f;
	//triangleList.m_Microfacets[4].alphaX = 0.0f;
	//triangleList.m_Microfacets[4].alphaY = 0.0f;

	//triangleList.addPlane(vec3(50.0f, -1.0f, 50.0f), vec3(-50.0f, 1.0f, 50.0f), vec3(50.0f, -1.0f, -50.0f), tmat);
	triangleList.loadModel(sponza, .1f, glm::translate(mat4(1.0f), vec3(0.0f, -10.0f, 0.0f)), -1, false);
	//triangleList.loadModel(teapot, .9f, glm::translate(glm::rotate(mat4(1.0f), glm::radians(90.0f), vec3(0, 1, 0)), vec3(0.0f, -3.0f, 0.0f)), tmat);

	//triangleList.loadModel(conference, 15.0f, glm::translate(mat4(1.0f), vec3(0.0f, 0.0f, 0.0f)));
	//triangleList.loadModel(nanosuit, 15.0f, glm::translate(mat4(1.0f), vec3(0.0f, 0.0f, 0.0f)));

	//Material m = Material::fresnel(vec3(1.f), 1.2f, vec3(0.0f, .7f, .7f), 0.000001f);
	//const auto dragMat = triangleList.addMaterial(m);
	//triangleList.loadModel(dragon, 10.0f, glm::translate(glm::rotate(glm::mat4(1.0f), glm::radians(-90.f), vec3(1, 0, 0)), vec3(-2.0f, .0f, -2.0f)), dragMat);

	//triangleList.m_BeckmannMicrofacets[dragMat].alphax = 0.0001f;
	//triangleList.m_BeckmannMicrofacets[dragMat].alphay = 0.0001f;
	//triangleList.loadModel(dragon, 1.f, glm::rotate(glm::translate(mat4(1.0f), vec3(0.0f, -2.7f, 0.0f)), glm::radians(-90.0f), vec3(1, 0, 0)), dragMat);

	params.camera = &camera;

	auto cudaRenderer = CudaRenderer(params.width, params.height);
	window.SetResizeCallback([&camera, &cudaRenderer](int width, int height) {
		camera.SetWidthHeight(width, height);
		params.width = width, params.height = height;
		rayBufferSize = width * height * 2;
		allocateBuffers();
		cudaRenderer.setDimensions(width, height);
	});

	if (triangleList.getPrimCount() <= 0 || triangleList.m_LightIndices.empty())
	{
		std::cout << "No primitives and/or lights, exiting." << std::endl;
		return 0;
	}

	bool shouldExit = false;
	bool* keys = new bool[512];
	memset(keys, 0, sizeof(bool) * 512);
	auto* bvh = new StaticBVHTree(&triangleList, BVHType::SAH, tPool);
	bvh->ConstructBVH();
	auto* mbvh = new MBVHTree(bvh);
	mbvh->ConstructBVH();

	window.SetEventCallback([&triangleList, &mbvh, &shouldExit, &camera, &keys](Event event) {
		switch (event.type) {
		case(CLOSED):
			shouldExit = true;
			break;
		case(MOUSE):
			if (event.state == MOUSE_MOVE)
			{
				mouseX = event.realX;
				mouseY = event.realY;
				if (keys[GLFW_MOUSE_BUTTON_RIGHT])
					camera.ProcessMouse(event.x, -event.y), mouseMoved = true;
			}
			else if (event.state == MOUSE_SCROLL)
				camera.ChangeFOV(camera.GetFOV() + event.y), mouseMoved = true;
			if (event.state == KEY_PRESSED)
			{
				keys[event.key] = true;
			}
			else if (event.state == KEY_RELEASED)
			{
				keys[event.key] = false;
			}
			break;
		case(KEY): {
			if (event.state == KEY_PRESSED)
				keys[event.key] = true;
			else if (event.state == KEY_RELEASED)
				keys[event.key] = false;
		}
		}
	});

	cudaMalloc(&params.gpuRays, rayBufferSize * sizeof(Ray));
	cudaMalloc(&params.gpuNextRays, rayBufferSize * sizeof(Ray));
	cudaMalloc(&params.gpuShadowRays, rayBufferSize * sizeof(ShadowRay));

	cudaMalloc(&params.gpuScene.gpuMbvhNodes, mbvh->m_FinalPtr * sizeof(MBVHNode));
	cudaMemcpy(params.gpuScene.gpuMbvhNodes, mbvh->m_Tree.data(), mbvh->m_FinalPtr * sizeof(MBVHNode), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.gpuBvhNodes, bvh->m_BVHPool.size() * sizeof(BVHNode));
	cudaMemcpy(params.gpuScene.gpuBvhNodes, bvh->m_BVHPool.data(), bvh->m_BVHPool.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.gpuPrimIndices, bvh->m_PrimitiveIndices.size() * sizeof(int));
	cudaMemcpy(params.gpuScene.gpuPrimIndices, bvh->m_PrimitiveIndices.data(), bvh->m_PrimitiveIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.gpuMaterials, triangleList.m_Materials.size() * sizeof(Material));
	cudaMemcpy(params.gpuScene.gpuMaterials, triangleList.m_Materials.data(), triangleList.m_Materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.vertices, triangleList.m_Vertices.size() * sizeof(vec3));
	cudaMemcpy(params.gpuScene.vertices, triangleList.m_Vertices.data(), triangleList.m_Vertices.size() * sizeof(vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.normals, triangleList.m_Normals.size() * sizeof(vec3));
	cudaMemcpy(params.gpuScene.normals, triangleList.m_Normals.data(), triangleList.m_Normals.size() * sizeof(vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.centerNormals, triangleList.m_CenterNormals.size() * sizeof(vec3));
	cudaMemcpy(params.gpuScene.centerNormals, triangleList.m_CenterNormals.data(), triangleList.m_CenterNormals.size() * sizeof(vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.texCoords, triangleList.m_TexCoords.size() * sizeof(vec2));
	cudaMemcpy(params.gpuScene.texCoords, triangleList.m_TexCoords.data(), triangleList.m_TexCoords.size() * sizeof(vec2), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.indices, triangleList.m_Indices.size() * sizeof(uvec3));
	cudaMemcpy(params.gpuScene.indices, triangleList.m_Indices.data(), triangleList.m_Indices.size() * sizeof(uvec3), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.lightIndices, triangleList.m_LightIndices.size() * sizeof(unsigned int));
	cudaMemcpy(params.gpuScene.lightIndices, triangleList.m_LightIndices.data(), triangleList.m_LightIndices.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.gpuMatIdxs, triangleList.m_MaterialIdxs.size() * sizeof(unsigned int));
	cudaMemcpy(params.gpuScene.gpuMatIdxs, triangleList.m_MaterialIdxs.data(), triangleList.m_MaterialIdxs.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.currentFrame, params.width * params.height * sizeof(vec4));
	cudaMemset(params.gpuScene.currentFrame, 0, params.width * params.height * sizeof(vec4));

	const auto textureBuffer = triangleList.createTextureBuffer();

	cudaMalloc(&params.gpuScene.gpuTexBuffer, textureBuffer.textureColors.size() * sizeof(vec4));
	cudaMemcpy(params.gpuScene.gpuTexBuffer, textureBuffer.textureColors.data(), textureBuffer.textureColors.size() * sizeof(vec4), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.gpuTexDims, textureBuffer.textureDims.size() * sizeof(uint));
	cudaMemcpy(params.gpuScene.gpuTexDims, textureBuffer.textureDims.data(), textureBuffer.textureDims.size() * sizeof(uint), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.gpuTexOffsets, textureBuffer.textureOffsets.size() * sizeof(uint));
	cudaMemcpy(params.gpuScene.gpuTexOffsets, textureBuffer.textureOffsets.data(), textureBuffer.textureOffsets.size() * sizeof(uint), cudaMemcpyHostToDevice);

	cudaMalloc(&params.gpuScene.microfacets, triangleList.m_Microfacets.size() * sizeof(microfacet::Microfacet));
	cudaMemcpy(params.gpuScene.microfacets, triangleList.m_Microfacets.data(), triangleList.m_Microfacets.size() * sizeof(microfacet::Microfacet), cudaMemcpyHostToDevice);

	params.gpuScene.lightCount = triangleList.m_LightIndices.size();

	cudaDeviceSynchronize();

	auto reset = []() {
		cudaMemset(params.gpuScene.currentFrame, 0, params.width * params.height * sizeof(vec4));
		cudaDeviceSynchronize();
		mouseMoved = false;
		samples = 0;
	};

	Timer t;
	while (!shouldExit)
	{
		const int maxBufferSize = params.width * params.height;
		const float elapsed = t.elapsed();
		t.reset();
		window.PollEvents();

		shouldExit = keys[GLFW_KEY_ESCAPE] || keys[GLFW_KEY_Q];
		if (keys[GLFW_KEY_LEFT_ALT] && keys[GLFW_KEY_ENTER])
			window.SwitchFullscreen();
		launchKernels(cudaRenderer.getCudaArray(), params, samples, rayBufferSize);
		samples++;

		ImGui::Begin("Settings");
		const bool checkbox = params.gpuScene.skyboxEnabled;
		ImGui::Checkbox("Skybox", &params.gpuScene.skyboxEnabled);
		if (checkbox != params.gpuScene.skyboxEnabled) reset();
		ImGui::DragFloat("Speed", &movementSpeed, 0.2f, 0.2f, 100.f);
		ImGui::DragFloat3("Position", glm::value_ptr(camera.m_Position));
		ImGui::DragInt("RayBufferSize", &rayBufferSize, 16384.f, 1024, maxBufferSize);
		rayBufferSize = min(rayBufferSize, maxBufferSize);
		ImGui::DragFloat("N Epsilon", &params.gpuScene.normalEpsilon, 0.00001f, 1e-8f, 1e-2f, "%.8f");
		ImGui::DragFloat("ShadowDist Epsilon", &params.gpuScene.distEpsilon, 0.00001f, 1e-8f, 1e-2f, "%.8f");
		ImGui::DragFloat("Triangle Epsilon", &params.gpuScene.triangleEpsilon, 0.00001f, 1e-8f, 1e-2f, "%.8f");

		using namespace microfacet;

		if (keys[GLFW_KEY_T])
			focusedMat = getMaterialAtPixel(mbvh->m_Tree.data(), mbvh->m_PrimitiveIndices.data(), triangleList, camera, mouseX, mouseY);

		Material* mat = std::get<0>(focusedMat);
		Microfacet* mfMat = std::get<1>(focusedMat);

		if (mat != nullptr && mfMat != nullptr)
		{
			ImGui::BeginChild("Material");

			const bool aV = mfMat->sampleVisibility;
			const float aX = mfMat->alphaX;
			const float aY = mfMat->alphaY;
			const char* type;
			if (mat->type == Light) type = "Light";
			else if (mat->type == Lambertian) type = "Lambertian";
			else if (mat->type == Fresnel) type = "Fresnel";
			else if (mat->type == Specular) type = "Specular";
			else type = "Other";

			static bool linkValues = false;
			ImGui::Text("Type: %s", type);
			ImGui::Text("Mat index: %i", focusedMatIdx);
			if (mat->type == Light)
			{
				const vec3 emOrg = mat->emission;
				ImGui::DragFloat3("Emission", glm::value_ptr(mat->emission), 0.1f, 0.0f, 1000.0f);
				if (!glm::all(glm::equal(emOrg, mat->emission)))
				{
					cudaMemcpy(&params.gpuScene.gpuMaterials[focusedMatIdx], mat, sizeof(Material), cudaMemcpyHostToDevice);
					reset();
				}
			}
			else
			{
				ImGui::Checkbox("Link values", &linkValues);
				ImGui::Checkbox("Sample Visibility", &mfMat->sampleVisibility);
				ImGui::DragFloat("Roughness X", &mfMat->alphaX, 0.001f, 1e-6f, 1.0f);
				ImGui::DragFloat("Roughness Y", &mfMat->alphaY, 0.001f, 1e-6f, 1.0f);

				const vec3 orgAlbedo = mat->albedo;
				const vec3 orgAbsorption = mat->absorption;

				if (mat->diffuseTex == -1)
					ImGui::DragFloat3("Color", glm::value_ptr(mat->albedo), 0.1f, 0.0f, 1.0f);
				if (mat->type == Fresnel)
					ImGui::DragFloat3("Absorption", glm::value_ptr(mat->absorption), 0.1f, 0.0f, 100.0f);

				if (!glm::all(glm::equal(orgAlbedo, mat->albedo)) || !glm::all(glm::equal(orgAbsorption, mat->absorption)))
				{
					cudaMemcpy(&params.gpuScene.gpuMaterials[focusedMatIdx], mat, sizeof(Material), cudaMemcpyHostToDevice);
					reset();
				}
			}

			if (aX != mfMat->alphaX)
			{
				if (linkValues) mfMat->alphaY = mfMat->alphaX;
				cudaMemcpy(&params.gpuScene.microfacets[focusedMatIdx], mfMat, sizeof(Microfacet), cudaMemcpyHostToDevice);
				reset();
			}
			else if (aY != mfMat->alphaY)
			{
				if (linkValues) mfMat->alphaX = mfMat->alphaY;
				cudaMemcpy(&params.gpuScene.microfacets[focusedMatIdx], mfMat, sizeof(Microfacet), cudaMemcpyHostToDevice);
				reset();
			}
			else if (aV != mfMat->sampleVisibility)
			{
				cudaMemcpy(&params.gpuScene.microfacets[focusedMatIdx], mfMat, sizeof(Microfacet), cudaMemcpyHostToDevice);
				reset();
			}

			ImGui::EndChild();
		}

		ImGui::End();
		cudaRenderer.draw();

		window.Present();

		if (camera.HandleKeys(keys, movementSpeed * elapsed) || mouseMoved || keys[GLFW_KEY_R])
			reset();
	}

	cudaFree(params.gpuRays);
	cudaFree(params.gpuNextRays);
	cudaFree(params.gpuShadowRays);
	cudaFree(params.gpuScene.gpuMaterials);
	cudaFree(params.gpuScene.gpuMbvhNodes);
	cudaFree(params.gpuScene.gpuPrimIndices);
	cudaFree(params.gpuScene.indices);
	cudaFree(params.gpuScene.gpuMatIdxs);
	cudaFree(params.gpuScene.vertices);
	cudaFree(params.gpuScene.normals);
	cudaFree(params.gpuScene.texCoords);
	cudaFree(params.gpuScene.currentFrame);
	cudaFree(params.gpuScene.gpuTexBuffer);
	cudaFree(params.gpuScene.gpuTexDims);
	cudaFree(params.gpuScene.gpuTexOffsets);

	delete bvh;
	delete mbvh;
	delete tPool;
	return 0;
}

void allocateBuffers()
{
	cudaDeviceSynchronize();
	cudaFree(params.gpuScene.currentFrame);
	cudaFree(params.gpuRays);
	cudaFree(params.gpuNextRays);
	cudaFree(params.gpuShadowRays);

	cuda(Malloc(&params.gpuRays, rayBufferSize * sizeof(Ray)));
	cuda(Malloc(&params.gpuNextRays, rayBufferSize * sizeof(Ray)));
	cuda(Malloc(&params.gpuShadowRays, rayBufferSize * sizeof(ShadowRay)));

	cuda(Malloc(&params.gpuScene.currentFrame, params.width * params.height * sizeof(vec4)));
	cuda(Memset(params.gpuScene.currentFrame, 0, params.width * params.height * sizeof(vec4)));
}

std::tuple<Material*, microfacet::Microfacet*> getMaterialAtPixel(const MBVHNode* nodes, const unsigned int* primIndices, TriangleList& tList, Camera& camera, int x, int y)
{
	Ray ray = camera.GenerateRay(float(x), float(y));
	MBVHNode::traverseMBVH(ray.origin, ray.direction, &ray.t, &ray.hit_idx, nodes, primIndices, tList);

	if (ray.valid())
	{
		focusedMatIdx = tList.m_MaterialIdxs[ray.hit_idx];
		return std::make_tuple(&tList.m_Materials[focusedMatIdx], &tList.m_Microfacets[focusedMatIdx]);
	}

	return std::make_tuple(nullptr, nullptr);
}