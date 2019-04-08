#pragma once

#define GLM_FORCE_PURE
#include <glm/glm.hpp>

using namespace glm;

enum MaterialType {
	None = -1,
	Light = 0,
	Lambertian = 1,
	Specular = 2,
	Fresnel = 3,
	Beckmann = 4,
	FresnelBeckmann = 5,
	GGX = 6,
	FresnelGGX = 7,
};

struct Material {
	union {
		vec3 albedo;
		vec3 emission;
	};
	float refractIdx;
	int diffuseTex = -1;
	int normalTex = -1;
	int maskTex = -1;
	int displaceTex = -1;
	MaterialType type;
	float roughness;
	vec3 absorption = vec3(0.0f);

	static Material light(vec3 emission)
	{
		Material mat;
		mat.refractIdx = 1.0f;
		mat.diffuseTex = -1;
		mat.normalTex = -1;
		mat.type = Light;
		mat.emission = glm::max(emission, vec3(0.01f));
		mat.absorption = vec3(0.0f);
		return mat;
	}

	static Material lambertian(vec3 albedo, int diffuseTex = -1, int normalTex = -1, int maskTex = -1, int displaceTex = -1, float refractionIdx = 1.0f)
	{
		Material mat;
		mat.albedo = albedo;
		mat.refractIdx = refractionIdx;
		mat.diffuseTex = diffuseTex;
		mat.normalTex = normalTex;
		mat.maskTex = maskTex;
		mat.displaceTex = displaceTex;
		mat.type = Lambertian;
		mat.roughness = 0.0f;
		return mat;
	}

	static Material specular(vec3 albedo, float refractionIdx)
	{
		Material mat;
		mat.albedo = albedo;
		mat.refractIdx = refractionIdx;
		mat.type = Specular;
		mat.roughness = 0.0001f;
		return mat;
	}

	static Material fresnel(vec3 albedo, float refractIdx, vec3 absorption = vec3(0.0f), float roughness = 0.0f, int diffuseTex = -1, int normalTex = -1)
	{
		Material mat;
		mat.albedo = albedo;
		mat.refractIdx = refractIdx;
		mat.diffuseTex = diffuseTex;
		mat.normalTex = normalTex;
		mat.type = Fresnel;
		mat.absorption = absorption;
		mat.roughness = roughness;
		return mat;
	}
};