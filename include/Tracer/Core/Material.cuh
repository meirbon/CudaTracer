#pragma once

#define GLM_FORCE_PURE
#include <glm/glm.hpp>

#include <string>
#include <vector>

#include <Tracer/Core/Microfacet.cuh>

using namespace glm;

enum MaterialType {
	None = -1,
	Light = 0,
	Specular = 1,
	Fresnel = 2,
	Lambertian = 3,

	Beckmann = 4,
	GGX = 5,
	Trowbridge = 6,

	FresnelBeckmann = 7,
	FresnelGGX = 8,
	FresnelTrowbridge = 9,
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

	void changeType(const std::string& t)
	{
		if (t == "Light")
			type = Light;
		else if (t == "Specular")
			type = Specular;
		else if (t == "Fresnel")
			type = Fresnel;
		else if (t == "Lambertian")
			type = Lambertian;
		else if (t == "Beckmann")
			type = Beckmann;
		else if (t == "GGX")
			type = GGX;
		else if (t == "Trowbridge")
			type = Trowbridge;
		else if (t == "FresnelBeckmann")
			type = FresnelBeckmann;
		else if (t == "FresnelGGX")
			type = FresnelGGX;
		else if (t == "FresnelTrowbridge")
			type = FresnelTrowbridge;
	}

	void changeType(MaterialType t)
	{
		this->type = t;
	}

	static const char* getTypeName(MaterialType type)
	{
		switch (type)
		{
		case(None):
			return "None";
		case(Light):
			return "Light";
		case(Specular):
			return "Specular";
		case(Fresnel):
			return "Fresnel";
		case(Lambertian):
			return "Lambertian";
		case(Beckmann):
			return "Beckmann";
		case(GGX):
			return "GGX";
		case(Trowbridge):
			return "Trowbridge";
		case(FresnelBeckmann):
			return "FresnelBeckmann";
		case(FresnelGGX):
			return "FresnelGGX";
		case(FresnelTrowbridge):
			return "FresnelTrowbridge";
		default:
			return "Unkown";
		}
	}

	static std::vector<const char*> getTypes()
	{
		return {
			"Light\0",
			"Specular\0",
			"Fresnel\0",
			"Lambertian\0",
			"Beckmann\0",
			"GGX\0",
			"Trowbridge\0",
			"FresnelBeckmann\0",
			"FresnelGGX\0",
			"FresnelTrowbridge\0"
		};
	}

	__device__ __host__ glm::vec3 sample(const microfacet::Microfacet & mf, const glm::vec3 & wiLocal, float r1, float r2) const
	{
		vec3 wmLocal{};
		switch (this->type)
		{
		case(Beckmann):
			return mf.sample_beckmann(wiLocal, r1, r2);
		case(GGX):
			return mf.sample_ggx(wiLocal, r1, r2);
		case(Trowbridge):
			return mf.sample_trowbridge_reitz(wiLocal, r1, r2);
		case(FresnelBeckmann):
			return mf.sample_beckmann(wiLocal, r1, r2);
		case(FresnelGGX):
			return mf.sample_ggx(wiLocal, r1, r2);
		case(FresnelTrowbridge):
			return mf.sample_trowbridge_reitz(wiLocal, r1, r2);
		default:
			return { 0.0f, 0.0f, 0.0f };
		}
	}

	__device__ __host__ float evaluate(const microfacet::Microfacet & mf, const glm::vec3 & wo, const glm::vec3 & wm, const glm::vec3 & wi) const
	{
		switch (this->type)
		{
		case(Beckmann):
			return mf.pdf_beckmann(wo, wm, wi);
		case(GGX):
			return mf.pdf_ggx(wo, wm, wi);
		case(Trowbridge):
			return mf.pdf_trowbridge_reitz(wo, wm, wi);
		case(FresnelBeckmann):
			return mf.pdf_beckmann(wo, wm, wi);
		case(FresnelGGX):
			return mf.pdf_ggx(wo, wm, wi);
		case(FresnelTrowbridge):
			return mf.pdf_trowbridge_reitz(wo, wm, wi);
		default:
			return 0.0f;
		}
	}
};