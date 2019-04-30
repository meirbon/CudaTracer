#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <cuda_runtime.h>

using namespace glm;

__device__ __host__ inline unsigned int RandomInt(unsigned int& seed)
{
	seed ^= seed << 13;
	seed ^= seed << 17;
	seed ^= seed << 5;
	return seed;
}

__device__ __host__ inline float RandomFloat(unsigned int& seed)
{
	return float(RandomInt(seed))* 2.3283064365387e-10f;
}

__device__ inline int RandomIntMax(unsigned int& seed, int max) {
	return int(RandomFloat(seed) * (max + 0.99999f));
}

// T = x/right vector, B = y/up vector, N = normal facing in z direction
__device__ __host__ inline void convertToLocalSpace(const vec3 & N, vec3 * T, vec3 * B)
{
	const vec3 W = glm::abs(N.x) > 0.99f ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
	*T = normalize(cross(N, W));
	*B = cross(N, *T);
}
__device__ __host__ inline vec3 localToWorld(const vec3 & sample, const vec3 & T, const vec3 & B, const vec3 & normal)
{
	return sample.x* T + sample.y * B + sample.z * normal;
}

__device__ __host__ inline vec3 sampleToWorld(const vec3 & sample, const vec3 & normal)
{
	vec3 T, B;
	convertToLocalSpace(normal, &T, &B);
	return localToWorld(sample, T, B, normal);
}

__device__ __host__ inline vec3 sampleHemisphere(float r1, float r2)
{
	const float sinTheta = sqrt(1.0f - r1 * r1);
	const float phi = 2.0f * glm::pi<float>() * r2;
	const float x = sinTheta * cos(phi);
	const float z = sinTheta * sin(phi);
	return { x, r1, z };
}

__device__ inline glm::vec2 ConcentricSampleDisk(const glm::vec2 & u) {
	//Map from [0,1] to [-1,1]
	glm::vec2 uOffset = 2.f * u - glm::vec2(1.0f);

	// Handle degeneracy at the origin
	if (uOffset.x == 0 && uOffset.y == 0)
		return glm::vec2(0, 0);

	// Apply concentric mapping to point
	float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
		r = uOffset.x;
		theta = glm::pi<float>() / 4.0f * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = glm::pi<float>() / 2.0f - glm::pi<float>() / 4.0f * (uOffset.x / uOffset.y);
	}
	return r * glm::vec2(std::cosf(theta), std::sinf(theta));
}

__device__ inline void computeOrthonormalBasisNaive(const glm::vec3 & w, glm::vec3 * u, glm::vec3 * v) {
	if (fabs(w.x) > .9) { /*If W is to close to X axis then pick Y*/
		*u = glm::vec3{ 0.0f, 1.0f, 0.0f };
	}
	else { /*Pick X axis*/
		*u = glm::vec3{ 1.0f, 0.0f, 0.0f };
	}
	*u = normalize(cross(*u, w));
	*v = cross(w, *u);
}

__device__ inline glm::vec2 concentricSampleDisk(const glm::vec2 & u) {
	//Map from [0,1] to [-1,1]
	glm::vec2 uOffset = 2.f * u - glm::vec2(1, 1);

	// Handle degeneracy at the origin
	if (uOffset.x == 0 && uOffset.y == 0)
		return glm::vec2(0, 0);

	// Apply concentric mapping to point
	float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
		r = uOffset.x;
		theta = glm::pi<float>() / 4 * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = glm::pi<float>() / 2.0f - glm::pi<float>() / 4.0f * (uOffset.x / uOffset.y);
	}
	return r * glm::vec2(std::cosf(theta), std::sinf(theta));
}

//Generate stratified sample of 2D [0,1]^2
__device__ inline glm::vec2 Random2DStratifiedSample(unsigned int& seed) {
	//Set the size of the pixel in strata.
	constexpr int width2D = 4;
	constexpr int height2D = 4;
	constexpr float pixelWidth = 1.0f / width2D;
	constexpr float pixelHeight = 1.0f / height2D;

	const int chosenStratum = RandomIntMax(seed, width2D * height2D);
	//Compute stratum X in [0, width-1] and Y in [0,height -1]
	const int stratumX = chosenStratum % width2D;
	const int stratumY = (chosenStratum / width2D) % height2D;

	//Now we split up the pixel into [stratumX,stratumY] pieces.
	//Let's get the width and height of this sample

	const float stratumXStart = pixelWidth * stratumX;
	const float stratumYStart = pixelHeight * stratumY;

	const float randomPointInStratumX = stratumXStart + (RandomFloat(seed) * pixelWidth);
	const float randomPointInStratumY = stratumYStart + (RandomFloat(seed) * pixelHeight);
	return glm::vec2(randomPointInStratumX, randomPointInStratumY);
}

__device__ inline glm::vec3 ortho(glm::vec3 v) {
	return abs(v.x) > abs(v.z) ? glm::vec3(-v.y, v.x, 0.0f)
		: glm::vec3(0.0f, -v.z, v.y);
}

__device__ inline glm::vec3 getConeSample(glm::vec3 dir, float extent, unsigned int& seed) {
	// Create orthogonal vector (fails for z,y = 0)
	dir = normalize(dir);
	glm::vec3 o1 = glm::normalize(ortho(dir));
	glm::vec3 o2 = glm::normalize(glm::cross(dir, o1));

	// Convert to spherical coordinates aligned to direction
	glm::vec2 r = {
		(RandomInt(seed) >> 16) / 65535.0f,
		(RandomInt(seed) >> 16) / 65535.0f
	};

	r.x = r.x * 2.f * glm::pi<float>();
	r.y = 1.0f - r.y * extent;

	float oneminus = sqrt(1.0f - r.y * r.y);
	return cos(r.x) * oneminus * o1 + sin(r.x) * oneminus * o2 + r.y * dir;
}