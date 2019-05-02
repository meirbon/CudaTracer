#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext.hpp>

#include "Tracer/Application.h"
#include "Tracer/Core/TriangleList.h"
#include "Tracer/Utils/GLFWWindow.h"

int main(int argc, char* argv[])
{
	constexpr int width = 1280;
	constexpr int height = 720;

	using namespace utils;
	using namespace core;
	using namespace glm;

	auto window = GLFWWindow("CudaTracer", 1280, 720, false, false);
	window.setVsync(false);

	auto triangleList = TriangleList();

	const std::string sky = "Models/envmap.hdr";

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

	const auto mat = triangleList.addMaterial(Material::light(vec3(10)));

	//triangleList.loadModel(sphere, 1.0f, glm::translate(mat4(1.0f), vec3(8.0f, 3.7f, 0.0f)), mat);
	triangleList.loadModel(sphere, 10.0f, glm::translate(mat4(1.0f), vec3(4.0f, 40.0f, 0.0f)), mat);
	//triangleList.loadModel(sphere, .1f, glm::translate(mat4(1.0f), vec3(0.0f, 10.f, 0.0f)), mat);
	//triangleList.loadModel(sphere, 1.0f, glm::translate(mat4(1.0f), vec3(-4.0f, 3.7f, 0.0f)), mat);
	//triangleList.loadModel(sphere, 1.0f, glm::translate(mat4(1.0f), vec3(-8.0f, 3.7f, 0.0f)), mat);

	const auto fMat = triangleList.addMaterial(Material::fresnel(vec3(1), 1.0f));
	auto tempMat = Material::lambertian(vec3(1.f, .8f, .8f));
	tempMat.type = Beckmann;
	const auto mMat = triangleList.addMaterial(tempMat);

	triangleList.loadModel(sphere, 3.0f, glm::translate(mat4(1.0f), vec3(-40.0f, 11.0f, -2.0f)), fMat);
	triangleList.loadModel(sphere, 3.0f, glm::translate(mat4(1.0f), vec3(-30.0f, 11.0f, -2.0f)), mMat);

	//triangleList.addPlane(vec3(50.0f, -1.0f, 50.0f), vec3(-50.0f, 1.0f, 50.0f), vec3(50.0f, -1.0f, -50.0f), tmat);
	triangleList.loadModel(sponza, .1f, glm::translate(mat4(1.0f), vec3(0.0f, -10.0f, 0.0f)), -1, false);
	//triangleList.loadModel(teapot, .9f, glm::translate(glm::rotate(mat4(1.0f), glm::radians(90.0f), vec3(0, 1, 0)), vec3(0.0f, -3.0f, 0.0f)), tmat);

	//triangleList.loadModel(conference, 15.0f, glm::translate(mat4(1.0f), vec3(0.0f, 0.0f, 0.0f)));
	//triangleList.loadModel(nanosuit, 15.0f, glm::translate(mat4(1.0f), vec3(0.0f, 0.0f, 0.0f)));

	Material m = Material::fresnel(vec3(1.f), 1.2f, vec3(0.0f, .7f, .7f), 0.000001f);
	const auto dragMat = triangleList.addMaterial(m);
	triangleList.loadModel(dragon, 10.0f, glm::translate(glm::rotate(glm::mat4(1.0f), glm::radians(-90.f), vec3(1, 0, 0)), vec3(-2.0f, .0f, -2.0f)), dragMat);

	//triangleList.m_BeckmannMicrofacets[dragMat].alphax = 0.0001f;
	//triangleList.m_BeckmannMicrofacets[dragMat].alphay = 0.0001f;
	//triangleList.loadModel(dragon, 1.f, glm::rotate(glm::translate(mat4(1.0f), vec3(0.0f, -2.7f, 0.0f)), glm::radians(-90.0f), vec3(1, 0, 0)), dragMat);

	auto application = Application(&window, &triangleList);

	application.loadSkybox(sky);
	application.run();
}
