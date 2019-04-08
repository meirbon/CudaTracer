#pragma once

#include <cuda_runtime.h>

#include <GL/glew.h>
#define GLM_FORCE_PURE
#include <glm/glm.hpp>

#include <cstdio>

#include "../Material.cuh"
#include "Random.cuh"

#define MAX_DISTANCE 1e34f

constexpr float EPSILON = 0.0001f;

using namespace glm;

struct Ray
{
	glm::vec3 origin;
	glm::vec3 direction;
	glm::vec3 throughput;
	MaterialType lastBounceType = None;
	glm::vec3 lastNormal;
	float t;
	int hit_idx;
	int index;
	int bounces = 0;

	__device__ Ray() = default;

	__device__ inline bool valid() const
	{
		return this->t < MAX_DISTANCE;
	}

	__device__ inline glm::vec3 getHitpoint() const
	{
		return origin + t * direction;
	}

	__device__  inline Ray(glm::vec3 org, glm::vec3 dir, int index)
	{
		this->origin = org;
		this->direction = dir;
		this->throughput = glm::vec3(1.0f);
		this->t = MAX_DISTANCE;
		this->hit_idx = -1;
		this->index = index;
		this->bounces = 0;
	}

	__device__ inline void reflect(const glm::vec3& normal)
	{
		direction = glm::reflect(direction, normal);
	}

	__device__ inline void reflectDiffuse(const glm::vec3& normal, float r1, float r2)
	{
		vec3 T, B;
		convertToLocalSpace(normal, &T, &B);
		const vec3 sample = sampleHemisphere(r1, r2);
		direction = normalize(localToWorld(sample, T, B, normal));
	}

	__device__ inline void reflectCosineWeighted(float r1, float r2)
	{
		const float r = sqrtf(r1);
		const float theta = 2.0f * glm::pi<float>() * r2;
		const float x = r * cos(theta);
		const float z = r * sin(theta);
		direction = { x, glm::max(0.0f, sqrtf(1.0f - r1)), z };
	}

	__device__ inline static Ray generate(const glm::vec3 &origin, const glm::vec3 &viewDir, const glm::vec3 &horizontal, const glm::vec3 &vertical, float x, float y, float invw, float invh, int index)
	{
		const float pixelX = float(x) * invw;
		const float pixelY = float(y) * invh;

		const float screenX = 2.0f * pixelX - 1.0f;
		const float screenY = 1.0f - 2.0f * pixelY;

		const glm::vec3 p = viewDir + horizontal * screenX + vertical * screenY;
		return {
			origin,
			glm::normalize(p),
			index
		};
	}
};

struct ShadowRay
{
	glm::vec3 origin;
	glm::vec3 direction;
	glm::vec3 color;
	float t;
	int index;

	__device__ inline ShadowRay(vec3 org, vec3 dir, vec3 col, float maxDist, int idx)
		: origin(org), direction(dir), color(col), t(maxDist), index(idx)
	{
	}

	__device__ inline void print()
	{
		printf("org: %f %f %f, dir: %f %f %f, col: %f %f %f, t: %f, index: %i\n", origin.x, origin.y, origin.z, direction.x, direction.y, direction.z, color.r, color.g, color.b, t, index);
	}
};