#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "Random.cuh"
#include "Tools.h"

struct ShadingData
{
	glm::vec3 color;
	float roughness;
	float eta;
};

enum BSDFType
{
	SPECULAR = 0,
	DIFFUSE = 1,
	TRANSMISION = 2
};

// for debugging: Lambert brdf
__device__ static glm::vec3 EvaluateBSDF(const ShadingData& data, const glm::vec3 iN, const glm::vec3 T, const glm::vec3 wi,
	const glm::vec3 wo, float& pdf)
{
	pdf = abs(dot(wo, iN)) * glm::one_over_pi<float>();
	return data.color * glm::one_over_pi<float>();
}

__device__ static glm::vec3 SampleBSDF(const ShadingData& data, const glm::vec3 iN, const glm::vec3 N, const glm::vec3 T,
	const glm::vec3 B, const glm::vec3 wi, const float r3, const float r4, glm::vec3& wo, float& pdf, BSDFType& reflectionType)
{
	if (data.roughness < 0.1f)
	{
		// pure specular
		wo = reflect(-wi, iN);

		if (dot(N, wo) <= 0.0f)
		{
			pdf = 0.0f;
			return glm::vec3(0.0f);
		}

		pdf = 1.0f;
		reflectionType = SPECULAR;
		return data.color * (1.0f / abs(dot(iN, wo)));
	}

	wo = glm::normalize(Tangent2World(DiffuseReflectionCosWeighted(r3, r4), iN));

	if (dot(N, wo) <= 0.0f)
	{
		pdf = 0.0f;
		return glm::vec3(0.0f);
	}

	reflectionType = DIFFUSE;
	pdf = glm::max(0.0f, dot(wo, iN)) * glm::one_over_pi<float>();
	return data.color * glm::one_over_pi<float>();
}

// EOF