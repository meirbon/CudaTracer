#include "kernel.cuh"

#include <cuda_runtime.h>
#include <surface_functions.h>
#include <cuda_surface_types.h>
#include <device_launch_parameters.h>

#include "cuda_definitions.h"
#include "cuda_assert.h"

#include "Core/SceneData.cuh"
#include "BVH/MBVHNode.cuh"
#include "BVH/BVHNode.cuh"
#include "Core/Random.cuh"

using namespace glm;

#ifdef __CUDACC__
#define LAUNCH_BOUNDS __launch_bounds__(128, 8)
#else
#define LAUNCH_BOUNDS
#endif

#define USE_MICROFACETS 1
#define MAX_DEPTH 16
#define PI glm::pi<float>()
#define INVPI glm::one_over_pi<float>()

__device__ int primary_ray_cnt = 0;
//The index of the ray at which we start generating more rays in ray generation step.
//Effectively is the last index which was previously generated + 1.
__device__ int start_position = 0;
//Ray number incremented by every thread in primary_rays ray generation
__device__ int ray_nr_primary = 0;
//Ray number to fetch different ray from every CUDA thread during the extend step.
__device__ int ray_nr_extend = 0;
//Ray number to fetch different ray from every CUDA thread in the shade step.
__device__ int ray_nr_shade = 0;
__device__ int ray_nr_light = 0;
__device__ int ray_nr_diffuse = 0;
//Ray number to fetch different ray from every CUDA thread in the connect step.
__device__ int ray_nr_connect = 0;
//Number of shadow rays generated in shade step, which are placed in connect step.
__device__ int shadow_ray_cnt = 0;

surface<void, cudaSurfaceType2D> framebuffer;

__device__ inline void draw(unsigned int x, unsigned int y, const vec4 &color)
{
	surf2Dwrite(color, framebuffer, x * sizeof(vec4), y);
}

__device__ inline void draw_unbounded(unsigned int x, unsigned int y, const vec4 &color)
{
	surf2Dwrite(color, framebuffer, x * sizeof(vec4), y, cudaBoundaryModeZero);
}

__global__ void setGlobals(int rayBufferSize, int width, int height)
{
	const int maxBuffer = width * height;
	const unsigned int progress = rayBufferSize - (glm::min(primary_ray_cnt, rayBufferSize));
	start_position += progress;
	start_position %= maxBuffer;

	shadow_ray_cnt = 0;
	primary_ray_cnt = 0;
	ray_nr_primary = 0;
	ray_nr_extend = 0;
	ray_nr_shade = 0;

	ray_nr_light = 0;
	ray_nr_diffuse = 0;

	ray_nr_connect = 0;
}

__global__ void generatePrimaryRays(
	Ray* rays,
	vec3 origin,
	vec3 viewDir,
	vec3 hor,
	vec3 ver,
	int w,
	int h,
	float invw,
	float invh,
	int rayBufferSize,
	unsigned int frame
)
{
	while (true)
	{
		const int index = atomicAggInc(&ray_nr_primary);
		//const int index = atomicAdd(&ray_nr_primary, 1);
		// Start from where extended rays ended
		const int rayIdx = index + primary_ray_cnt;
		if (rayIdx >= rayBufferSize) return;

		unsigned int seed = (index + frame * 147565741) * 720898027 * index;

		const int x = (start_position + index) % w;
		const int y = ((start_position + index) / w) % h;

		const float px = float(x) + RandomFloat(seed) - 0.5f;
		const float py = float(y) + RandomFloat(seed) - 0.5f;

		rays[rayIdx] = Ray::generate(origin, viewDir, hor, ver, px, py, invw, invh, x + y * w);
	}
}

__global__ void LAUNCH_BOUNDS extend(Ray* rays, SceneData scene, int rayBufferSize)
{
	while (true)
	{
		const int index = atomicAdd(&ray_nr_extend, 1);

		if (index >= rayBufferSize) return;

		Ray& ray = rays[index];
		ray.t = MAX_DISTANCE;
		MBVHNode::traverseMBVH(ray.origin, ray.direction, &ray.t, &ray.hit_idx, scene);
	}
}

__global__ void LAUNCH_BOUNDS shade_invalid(Ray* rays, Ray* eRays, ShadowRay* sRays, SceneData scene, unsigned int frame, int rayBufferSize)
{
	while (true)
	{
		const int index = atomicAdd(&ray_nr_light, 1);
		if (index >= rayBufferSize) return;

		Ray& ray = rays[index];
		vec3 color = vec3(0.0f);
		float alpha = 1.0f;
		if (ray.valid())
		{
			const Material &mat = scene.gpuMaterials[scene.gpuMatIdxs[ray.hit_idx]];
			if (mat.type != Light) continue;

			unsigned int seed = (frame * ray.index * 147565741) * 720898027 * index;
			int new_frame = 0;

			ray.origin = ray.getHitpoint();
			const uvec3 tIdx = scene.indices[ray.hit_idx];
			const vec3 cN = scene.centerNormals[ray.hit_idx];
			const vec3 bary = triangle::getBaryCoords(ray.origin, cN, scene.vertices[tIdx.x], scene.vertices[tIdx.y], scene.vertices[tIdx.z]);

			const vec2 tCoords = triangle::getTexCoords(bary, scene.texCoords[tIdx.x], scene.texCoords[tIdx.y], scene.texCoords[tIdx.z]);
			vec3 normal;
			if (mat.normalTex >= 0)
			{
				vec3 T, B;
				convertToLocalSpace(cN, &T, &B);
				const vec3 n = scene.getTextureNormal(mat.normalTex, tCoords);
				normal = normalize(localToWorld(n, T, B, cN));
			}
			else
				normal = triangle::getNormal(bary, scene.normals[tIdx.x], scene.normals[tIdx.y], scene.normals[tIdx.z]);

			const bool backFacing = glm::dot(normal, ray.direction) >= 0.0f;
			if (backFacing) normal *= -1.0f;

			const vec3 matColor = mat.diffuseTex < 0 ? mat.albedo : scene.getTextureColor(mat.diffuseTex, tCoords);

			const int matIdx = scene.gpuMatIdxs[ray.hit_idx];
			const auto mf = scene.microfacets[scene.gpuMatIdxs[ray.hit_idx]];

			if (ray.bounces <= 0)
				color = mat.emission;
			else if (ray.lastBounceType == Fresnel)
				color = ray.throughput * mat.emission;
			else
			{
				const float NdotL = dot(ray.lastNormal, ray.direction);
				const float LNdotL = dot(normal, -ray.direction);
				const float lightPDF = ray.t * ray.t / (LNdotL * triangle::getArea(scene.vertices[tIdx.x], scene.vertices[tIdx.y], scene.vertices[tIdx.z]));

				const float mfPDF = 1.0f / mf.pdf_trowbridge_reitz(ray.direction, ray.lastNormal, glm::reflect(-ray.direction, ray.lastNormal));
				const vec3 col = ray.throughput * mat.emission * float(scene.lightCount) * NdotL;
				if (lightPDF > 0 && mfPDF > 0)
				{
					const float sum = mfPDF + lightPDF;
					const float w1 = mfPDF / sum;
					const float w2 = lightPDF / sum;
					const float pdf = max(0.0f, 1.0f / (w1 * mfPDF + w2 * lightPDF));
					color = col * pdf;
				}
				else
				{
					alpha = 0.0f;
				}
			}
		}
		else
		{
			const vec2 uv = {
				(1.0f + atan2f(ray.direction.x, -ray.direction.z) / glm::pi<float>()) / 2.0f,
				1.0f - acosf(ray.direction.y) / glm::pi<float>()
			};

			color = scene.skyboxEnabled ? ray.throughput * vec3(scene.getTextureColor(scene.skyboxTexture, uv)) : vec3(0.0f);
		}

		ray.throughput = vec3(0.0f);
		atomicAdd(&scene.currentFrame[ray.index].r, color.r);
		atomicAdd(&scene.currentFrame[ray.index].g, color.g);
		atomicAdd(&scene.currentFrame[ray.index].b, color.b);
		atomicAdd(&scene.currentFrame[ray.index].a, alpha);
	}
}

__global__ void LAUNCH_BOUNDS shade_microfacet(Ray* rays, Ray* eRays, ShadowRay* sRays, SceneData scene, unsigned int frame, int rayBufferSize)
{
	while (true)
	{
		const int index = atomicAdd(&ray_nr_diffuse, 1);
		if (index >= rayBufferSize) return;

		Ray& ray = rays[index];
		if (!ray.valid()) continue;

		const Material &mat = scene.gpuMaterials[scene.gpuMatIdxs[ray.hit_idx]];
		if (mat.type == Light) continue;

		vec3 color = vec3(0.0f);
		unsigned int seed = (frame * ray.index * 147565741) * 720898027 * index;

		ray.origin = ray.getHitpoint();
		const uvec3 tIdx = scene.indices[ray.hit_idx];
		const vec3 cN = scene.centerNormals[ray.hit_idx];
		const vec3 bary = triangle::getBaryCoords(ray.origin, cN, scene.vertices[tIdx.x], scene.vertices[tIdx.y], scene.vertices[tIdx.z]);
		const vec2 tCoords = triangle::getTexCoords(bary, scene.texCoords[tIdx.x], scene.texCoords[tIdx.y], scene.texCoords[tIdx.z]);
		vec3 normal;

		if (mat.normalTex >= 0)
		{
			vec3 T, B;
			convertToLocalSpace(cN, &T, &B);
			const vec3 n = scene.getTextureNormal(mat.normalTex, tCoords);
			normal = normalize(localToWorld(n, T, B, cN));
		}
		else
		{
			normal = triangle::getNormal(bary, scene.normals[tIdx.x], scene.normals[tIdx.y], scene.normals[tIdx.z]);
		}

		const bool backFacing = glm::dot(normal, ray.direction) >= 0.0f;
		normal *= backFacing ? -1.0f : 1.0f;

		const vec3 matColor = mat.diffuseTex < 0 ? mat.albedo : scene.getTextureColor(mat.diffuseTex, tCoords);
		const auto mf = scene.microfacets[scene.gpuMatIdxs[ray.hit_idx]];

		const vec3 wi = -ray.direction;

		vec3 T, B;
		convertToLocalSpace(normal, &T, &B);
		const vec3 wiLocal = normalize(vec3(dot(T, wi), dot(B, wi), dot(normal, wi)));

		// Local half-way vector
		const vec3 wmLocal = mf.sample_trowbridge_reitz(wiLocal, RandomFloat(seed), RandomFloat(seed));
		// Half-way vector
		const vec3 wm = T * wmLocal.x + B * wmLocal.y + normal * wmLocal.z;
		// Local new ray direction
		const vec3 woLocal = glm::reflect(-wiLocal, wmLocal); // Reflect

		// New outgoing ray direction
		const vec3 wo = localToWorld(woLocal, T, B, wm);
		const float PDF = mf.pdf_trowbridge_reitz(woLocal, wiLocal, wmLocal);
		ray.origin += wm * scene.normalEpsilon;

		const int light = RandomIntMax(seed, scene.lightCount - 1);
		const uvec3 lightIdx = scene.indices[scene.lightIndices[light]];
		const vec3 lightPos = triangle::getRandomPointOnSurface(scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z], RandomFloat(seed), RandomFloat(seed));
		vec3 L = lightPos - ray.origin;
		const float squaredDistance = dot(L, L);
		const float distance = sqrtf(squaredDistance);
		L /= distance;

		const vec3 cNormal = scene.centerNormals[scene.lightIndices[light]];
		const vec3 baryLight = triangle::getBaryCoords(lightPos, cNormal, scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z]);
		const vec3 lightNormal = triangle::getNormal(bary, scene.normals[lightIdx.x], scene.normals[lightIdx.y], scene.normals[lightIdx.z]);

		const float NdotL = dot(normal, L);
		const float LNdotL = dot(lightNormal, -L);

		if (NdotL > 0 && LNdotL > 0)
		{
			const float area = triangle::getArea(scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z]);

			const auto emission = scene.gpuMaterials[scene.gpuMatIdxs[light]].emission;
			const vec3 shadowCol = ray.throughput * matColor * emission * NdotL * float(scene.lightCount);

			const float mfPDF = 1.0f / mf.pdf_trowbridge_reitz(L, wi, wm);
			const float lightPDF = squaredDistance / (LNdotL * area);

			if (lightPDF > 0 && mfPDF > 0)
			{
				const unsigned int shadowIdx = atomicAdd(&shadow_ray_cnt, 1);

				const float sum = mfPDF + lightPDF;
				const float w1 = mfPDF / sum;
				const float w2 = lightPDF / sum;
				const float pdf = max(0.0f, 1.0f / (w1 * mfPDF + w2 * lightPDF));
				sRays[shadowIdx] = ShadowRay(
					ray.origin, std::move(L), shadowCol * pdf,
					distance - scene.distEpsilon, ray.index
				);
			}
		}

		ray.lastNormal = wm;
		ray.throughput *= matColor * PDF;
		ray.direction = wo;

		ray.throughput = glm::max(vec3(0.0f), ray.throughput);

		const float prob = glm::min(1.0f, glm::max(ray.throughput.x, glm::min(ray.throughput.y, ray.throughput.z)));
		if (ray.bounces < MAX_DEPTH && prob > EPSILON && prob > RandomFloat(seed))
		{
			ray.bounces++;
			ray.throughput /= prob;

			unsigned int primary_index = atomicAdd(&primary_ray_cnt, 1);
			ray.lastBounceType = mat.type;
			eRays[primary_index] = ray;
		}
		else
		{
			ray.throughput = vec3(0.0f);
			atomicAdd(&scene.currentFrame[ray.index].a, 1.0f);
		}
	}
}

__global__ void LAUNCH_BOUNDS shade_lights(Ray* rays, Ray* eRays, ShadowRay* sRays, SceneData scene, unsigned int frame, int rayBufferSize)
{
	while (true)
	{
		const int index = atomicAdd(&ray_nr_light, 1);
		if (index >= rayBufferSize) return;

		Ray& ray = rays[index];
		if (!ray.valid()) continue;
		const Material &mat = scene.gpuMaterials[scene.gpuMatIdxs[ray.hit_idx]];
		if (mat.type != Light) continue;

		vec3 color = vec3(0.0f);
		unsigned int seed = (frame * ray.index * 147565741) * 720898027 * index;
		int new_frame = 0;

		ray.origin = ray.getHitpoint();
		const uvec3 tIdx = scene.indices[ray.hit_idx];
		const vec3 cN = scene.centerNormals[ray.hit_idx];
		const vec3 bary = triangle::getBaryCoords(ray.origin, cN, scene.vertices[tIdx.x], scene.vertices[tIdx.y], scene.vertices[tIdx.z]);

		const vec2 tCoords = triangle::getTexCoords(bary, scene.texCoords[tIdx.x], scene.texCoords[tIdx.y], scene.texCoords[tIdx.z]);
		vec3 normal;
		if (mat.normalTex >= 0)
		{
			vec3 T, B;
			convertToLocalSpace(cN, &T, &B);
			const vec3 n = scene.getTextureNormal(mat.normalTex, tCoords);
			normal = normalize(localToWorld(n, T, B, cN));
		}
		else
			normal = triangle::getNormal(bary, scene.normals[tIdx.x], scene.normals[tIdx.y], scene.normals[tIdx.z]);

		const bool backFacing = glm::dot(normal, ray.direction) >= 0.0f;
		if (backFacing) normal *= -1.0f;

		const vec3 matColor = mat.diffuseTex < 0 ? mat.albedo : scene.getTextureColor(mat.diffuseTex, tCoords);

		const int matIdx = scene.gpuMatIdxs[ray.hit_idx];
		const auto mf = scene.microfacets[scene.gpuMatIdxs[ray.hit_idx]];

		if (ray.bounces <= 0)
			color = mat.emission;
		else if (ray.lastBounceType == Fresnel)
			color = ray.throughput * mat.emission;
		else
		{
			const float NdotL = dot(ray.lastNormal, ray.direction);
			const float LNdotL = dot(normal, -ray.direction);
			const float lightPDF = ray.t * ray.t / (LNdotL * triangle::getArea(scene.vertices[tIdx.x], scene.vertices[tIdx.y], scene.vertices[tIdx.z]));

			const float mfPDF = 1.0f / mf.pdf_trowbridge_reitz(ray.direction, ray.lastNormal, glm::reflect(-ray.direction, ray.lastNormal));
			const vec3 col = ray.throughput * mat.emission * float(scene.lightCount) * NdotL;
			if (lightPDF > 0 && mfPDF > 0)
			{
				const float sum = mfPDF + lightPDF;
				const float w1 = mfPDF / sum;
				const float w2 = lightPDF / sum;
				const float pdf = max(0.0f, 1.0f / (w1 * mfPDF + w2 * lightPDF));
				color = col * pdf;
			}
		}

		ray.throughput = vec3(0.0f);

		atomicAdd(&scene.currentFrame[ray.index].r, color.r);
		atomicAdd(&scene.currentFrame[ray.index].g, color.g);
		atomicAdd(&scene.currentFrame[ray.index].b, color.b);
		atomicAdd(&scene.currentFrame[ray.index].a, 1.0f);
	}
}

__global__ void LAUNCH_BOUNDS shade(Ray* rays, Ray* eRays, ShadowRay* sRays, SceneData scene, unsigned int frame, int rayBufferSize)
{
	while (true)
	{
		const int index = atomicAggInc(&ray_nr_shade);
		/*const int index = atomicAdd(&ray_nr_shade, 1);*/
		if (index >= rayBufferSize) return;

		Ray& ray = rays[index];
		vec3 color = vec3(0.0f);
		unsigned int seed = (frame * ray.index * 147565741) * 720898027 * index;
		int new_frame = 0;

		if (!ray.valid())
		{
			const vec2 uv = {
				(1.0f + atan2f(ray.direction.x, -ray.direction.z) / glm::pi<float>()) / 2.0f,
				1.0f - acosf(ray.direction.y) / glm::pi<float>()
			};

			if (scene.skyboxEnabled)
				color = ray.throughput * vec3(scene.getTextureColor(scene.skyboxTexture, uv));

			ray.throughput = vec3(0.0f);
			new_frame++;
		}
		else
		{
			ray.origin = ray.getHitpoint();
			const uvec3 tIdx = scene.indices[ray.hit_idx];
			const vec3 cN = scene.centerNormals[ray.hit_idx];
			const vec3 bary = triangle::getBaryCoords(ray.origin, cN, scene.vertices[tIdx.x], scene.vertices[tIdx.y], scene.vertices[tIdx.z]);
			const vec2 tCoords = triangle::getTexCoords(bary, scene.texCoords[tIdx.x], scene.texCoords[tIdx.y], scene.texCoords[tIdx.z]);
			vec3 normal;

			const Material &mat = scene.gpuMaterials[scene.gpuMatIdxs[ray.hit_idx]];

			if (mat.normalTex >= 0)
			{
				vec3 T, B;
				convertToLocalSpace(cN, &T, &B);
				const vec3 n = scene.getTextureNormal(mat.normalTex, tCoords);
				normal = normalize(localToWorld(n, T, B, cN));
			}
			else
			{
				normal = triangle::getNormal(bary, scene.normals[tIdx.x], scene.normals[tIdx.y], scene.normals[tIdx.z]);
			}

			const bool backFacing = glm::dot(normal, ray.direction) >= 0.0f;
			if (backFacing) normal *= -1.0f;

			const vec3 matColor = mat.diffuseTex < 0 ? mat.albedo : scene.getTextureColor(mat.diffuseTex, tCoords);

#if USE_MICROFACETS
			const int matIdx = scene.gpuMatIdxs[ray.hit_idx];
			const auto mf = scene.microfacets[scene.gpuMatIdxs[ray.hit_idx]];

			const vec3 wi = -ray.direction;

			vec3 T, B;
			convertToLocalSpace(normal, &T, &B);
			const vec3 wiLocal = normalize(vec3(dot(T, wi), dot(B, wi), dot(normal, wi)));

			// Local half-way vector
			const vec3 wmLocal = mf.sample_trowbridge_reitz(wiLocal, RandomFloat(seed), RandomFloat(seed));
			// Half-way vector
			const vec3 wm = T * wmLocal.x + B * wmLocal.y + normal * wmLocal.z;
			// Local new ray direction
			const vec3 woLocal = glm::reflect(-wiLocal, wmLocal); // Reflect

			switch (mat.type)
			{
			case Light: {
				if (ray.bounces <= 0)
					color = mat.emission;
				else if (ray.lastBounceType == Fresnel)
					color = ray.throughput * mat.emission;
				else
				{
					const float NdotL = dot(ray.lastNormal, ray.direction);
					const float LNdotL = dot(normal, -ray.direction);
					const float lightPDF = ray.t * ray.t / (LNdotL * triangle::getArea(scene.vertices[tIdx.x], scene.vertices[tIdx.y], scene.vertices[tIdx.z]));

					const float mfPDF = 1.0f / mf.pdf_trowbridge_reitz(ray.direction, ray.lastNormal, glm::reflect(-ray.direction, ray.lastNormal));
					const vec3 col = ray.throughput * mat.emission * float(scene.lightCount) * NdotL;
					if (lightPDF > 0 && mfPDF > 0)
					{
						const float sum = mfPDF + lightPDF;
						const float w1 = mfPDF / sum;
						const float w2 = lightPDF / sum;
						const float pdf = max(0.0f, 1.0f / (w1 * mfPDF + w2 * lightPDF));
						color = col * pdf;
					}
				}

				ray.throughput = vec3(0.0f);
				break;
			}
			case Specular: {
				ray.reflect(normal);
				ray.throughput *= matColor;
				ray.origin += wm * scene.normalEpsilon;

				break;
			}
			case Fresnel: {
				const float n1 = backFacing ? mat.refractIdx : 1.0f;
				const float n2 = backFacing ? 1.0f : mat.refractIdx;
				const float n = n1 / n2;
				const float cosTheta = dot(wi, wm);
				const float k = 1.0f - (n * n) * (1.0f - cosTheta * cosTheta);
				ray.origin += wm * scene.normalEpsilon;

				vec3 woL = std::move(woLocal);
				ray.lastBounceType = Specular;
				if (k > 0.0f)
				{
					const float a = n1 - n2;
					const float b = n1 + n2;
					const float R0 = (a * a) / (b * b);
					const float c = 1.0f - cosTheta;
					const float Fr = max(R0 + (1.0f - R0) * (c * c * c * c *c), 1e-4f);
					const float P = 0.25 + 0.5 * Fr;

					if (RandomFloat(seed) > Fr)
					{
						ray.lastBounceType = Fresnel;
						if (backFacing)
							ray.throughput *= exp(-mat.absorption * ray.t);
						ray.origin -= 2.0f * wm * scene.normalEpsilon;
						woL = normalize(n * -wiLocal + wmLocal * (n * cosTheta - sqrtf(k)));
						ray.throughput /= ((1.0f - Fr) / (1.0f - P));
					}
				}

				if (ray.lastBounceType == Specular)
				{
					const int light = RandomIntMax(seed, scene.lightCount - 1);
					const uvec3 lightIdx = scene.indices[scene.lightIndices[light]];
					const vec3 lightPos = triangle::getRandomPointOnSurface(scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z], RandomFloat(seed), RandomFloat(seed));
					vec3 L = lightPos - ray.origin;
					const float squaredDistance = dot(L, L);
					const float distance = sqrtf(squaredDistance);
					L /= distance;

					const vec3 cNormal = scene.centerNormals[scene.lightIndices[light]];
					const vec3 baryLight = triangle::getBaryCoords(lightPos, cNormal, scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z]);
					const vec3 lightNormal = triangle::getNormal(bary, scene.normals[lightIdx.x], scene.normals[lightIdx.y], scene.normals[lightIdx.z]);

					const float NdotL = dot(normal, L);
					const float LNdotL = dot(lightNormal, -L);

					if (NdotL > 0 && LNdotL > 0)
					{
						const float area = triangle::getArea(scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z]);
						const float solidAngle = LNdotL * area / squaredDistance;

						const float mfPDF = 1.0f / mf.pdf_trowbridge_reitz(L, wi, wm);
						const float lightPDF = 1.0f / solidAngle;

						const auto emission = scene.gpuMaterials[scene.gpuMatIdxs[light]].emission;
						const vec3 shadowCol = ray.throughput * matColor * emission * NdotL * float(scene.lightCount);

						if (lightPDF > 0 && mfPDF > 0)
						{
							const unsigned int shadowIdx = atomicAdd(&shadow_ray_cnt, 1);

							const float sum = mfPDF + lightPDF;
							const float w1 = mfPDF / sum;
							const float w2 = lightPDF / sum;
							sRays[shadowIdx] = ShadowRay(
								ray.origin, L, shadowCol / (w1 * mfPDF + w2 * lightPDF),
								distance - scene.distEpsilon, ray.index
							);

							ray.lastNormal = wm;
						}
					}
				}

				ray.direction = localToWorld(woL, T, B, wm);
				ray.throughput *= matColor;

				if (ray.lastBounceType != Fresnel)
					ray.throughput *= mf.pdf_trowbridge_reitz(woL, wiLocal, wmLocal);
				break;
			}
			default: {
				// New outgoing ray direction
				const vec3 wo = localToWorld(woLocal, T, B, wm);
				const float PDF = mf.pdf_trowbridge_reitz(woLocal, wiLocal, wmLocal);
				ray.origin += wm * scene.normalEpsilon;

				const int light = RandomIntMax(seed, scene.lightCount - 1);
				const uvec3 lightIdx = scene.indices[scene.lightIndices[light]];
				const vec3 lightPos = triangle::getRandomPointOnSurface(scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z], RandomFloat(seed), RandomFloat(seed));
				vec3 L = lightPos - ray.origin;
				const float squaredDistance = dot(L, L);
				const float distance = sqrtf(squaredDistance);
				L /= distance;

				const vec3 cNormal = scene.centerNormals[scene.lightIndices[light]];
				const vec3 baryLight = triangle::getBaryCoords(lightPos, cNormal, scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z]);
				const vec3 lightNormal = triangle::getNormal(bary, scene.normals[lightIdx.x], scene.normals[lightIdx.y], scene.normals[lightIdx.z]);

				const float NdotL = dot(normal, L);
				const float LNdotL = dot(lightNormal, -L);

				if (NdotL > 0 && LNdotL > 0)
				{
					const float area = triangle::getArea(scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z]);

					const auto emission = scene.gpuMaterials[scene.gpuMatIdxs[light]].emission;
					const vec3 shadowCol = ray.throughput * matColor * emission * NdotL * float(scene.lightCount);

					const float mfPDF = 1.0f / mf.pdf_trowbridge_reitz(L, wi, wm);
					const float lightPDF = squaredDistance / (LNdotL * area);

					if (lightPDF > 0 && mfPDF > 0)
					{
						const unsigned int shadowIdx = atomicAdd(&shadow_ray_cnt, 1);

						const float sum = mfPDF + lightPDF;
						const float w1 = mfPDF / sum;
						const float w2 = lightPDF / sum;
						const float pdf = max(0.0f, 1.0f / (w1 * mfPDF + w2 * lightPDF));
						sRays[shadowIdx] = ShadowRay(
							ray.origin, std::move(L), shadowCol * pdf,
							distance - scene.distEpsilon, ray.index
						);
					}
				}

				ray.lastNormal = wm;
				ray.throughput *= matColor * PDF;
				ray.direction = wo;
				break;
			}
			}

			ray.lastNormal = wm;
#else
			ray.origin += normal * EPSILON;

			switch (mat.type)
			{
			case Light: {
				if (ray.bounces <= 0)
					color = mat.emission;
				else if (ray.lastBounceType == Lambertian)// Multiple importance sampling
				{
					const float NdotL = dot(ray.lastNormal, ray.direction);
					const float LNdotL = dot(normal, -ray.direction);
					const float lightPDF = ray.t * ray.t / (LNdotL * triangle::getArea(scene.vertices[tIdx.x], scene.vertices[tIdx.y], scene.vertices[tIdx.z]));
					const float lambertPDF = dot(ray.lastNormal, ray.direction) * glm::one_over_pi<float>();
					const vec3 col = ray.throughput * mat.emission * float(scene.lightCount);
					if (lightPDF > 0 && lambertPDF > 0)
					{
						const float sum = lambertPDF + lightPDF;
						const float w1 = lambertPDF / sum;
						const float w2 = lightPDF / sum;
						color = col / (w1 * lambertPDF + w2 * lightPDF);
					}
				}
				else
					color = ray.throughput * mat.emission;

				ray.throughput = vec3(0.0f);
				break;
			}
			case Lambertian: {
				const int light = RandomIntMax(seed, scene.lightCount - 1);
				const uvec3 lightIdx = scene.indices[scene.lightIndices[light]];
				const vec3 lightPos = triangle::getRandomPointOnSurface(scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z], RandomFloat(seed), RandomFloat(seed));
				vec3 L = lightPos - ray.origin;
				const float squaredDistance = dot(L, L);
				const float distance = sqrtf(squaredDistance);
				L /= distance;

				const vec3 cNormal = scene.centerNormals[scene.lightIndices[light]];
				const vec3 baryLight = triangle::getBaryCoords(lightPos, cNormal, scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z]);
				const vec3 lightNormal = triangle::getNormal(bary, scene.normals[lightIdx.x], scene.normals[lightIdx.y], scene.normals[lightIdx.z]);

				const float NdotL = dot(normal, L);
				const float LNdotL = dot(lightNormal, -L);
				const vec3 BRDF = matColor * glm::one_over_pi<float>();

				if (NdotL > 0 && LNdotL > 0)
				{
					const float area = triangle::getArea(scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z]);
					const float solidAngle = LNdotL * area / squaredDistance;

					const auto emission = scene.gpuMaterials[scene.gpuMatIdxs[light]].emission;
					const vec3 shadowCol = ray.throughput * BRDF * emission * NdotL * float(scene.lightCount);

					const float lambertPDF = 1.0f / NdotL * glm::one_over_pi<float>();
					const float lightPDF = 1.0f / solidAngle;

					if (lightPDF > 0 && lambertPDF > 0)
					{
						const unsigned int shadowIdx = atomicAdd(&shadow_ray_cnt, 1);

						const float sum = lambertPDF + lightPDF;
						const float w1 = lambertPDF / sum;
						const float w2 = lightPDF / sum;
						sRays[shadowIdx] = ShadowRay(
							ray.origin, std::move(L), shadowCol / (w1 * lambertPDF + w2 * lightPDF),
							distance - EPSILON, ray.index
						);
						ray.lastNormal = normal;
					}
				}

				ray.reflectCosineWeighted(RandomFloat(seed), RandomFloat(seed));
				const float NdotR = dot(normal, ray.direction);
				const float PDF = NdotR * glm::one_over_pi<float>();
				ray.lastBounceType = Lambertian;
				ray.throughput *= BRDF * NdotR / PDF;
				break;
			}
			case Specular: {
				ray.throughput *= matColor;
				ray.reflect(normal);
				ray.lastBounceType = Specular;
				break;
			}
			case Fresnel: {
				ray.throughput *= matColor;
				const vec3 dir = ray.direction;
				ray.reflect(normal);
				ray.lastBounceType = Specular;

				const float n1 = backFacing ? mat.refractIdx : 1.0f;
				const float n2 = backFacing ? 1.0f : mat.refractIdx;
				const float n = n1 / n2;
				const float cosTheta = dot(normal, -dir);
				const float k = 1.0f - (n * n) * (1.0f - cosTheta * cosTheta);

				if (k <= 0) break;

				const float a = n1 - n2;
				const float b = n1 + n2;
				const float R0 = (a * a) / (b * b);
				const float c = 1.0f - cosTheta;
				const float Fr = R0 + (1.0f - R0) * (c * c * c * c *c);

				const float r = RandomFloat(seed);
				if (r > Fr)
				{
					ray.lastBounceType = Fresnel;
					if (backFacing)
						ray.throughput *= exp(-mat.absorption * ray.t);;
					ray.origin -= 2.0f * normal;
					ray.direction = normalize(n * ray.direction + normal * (n * cosTheta - sqrtf(k)));
				}
				break;
			}
			default:
				break;
			}
#endif
			ray.throughput = glm::max(vec3(0.0f), ray.throughput);

			const float prob = glm::min(1.0f, glm::max(ray.throughput.x, glm::min(ray.throughput.y, ray.throughput.z)));
			if (ray.bounces < MAX_DEPTH && prob > EPSILON && prob > RandomFloat(seed))
			{
				ray.bounces++;
				ray.throughput /= prob;

				unsigned int primary_index = atomicAdd(&primary_ray_cnt, 1);
				ray.lastBounceType = mat.type;
				eRays[primary_index] = ray;
			}
			else
			{
				ray.throughput = vec3(0.0f);
				new_frame++;
			}
		}

		const float length = glm::length(color);
		if (length > 10.0f)
			color = color / length * 10.0f;

		atomicAdd(&scene.currentFrame[ray.index].r, color.r);
		atomicAdd(&scene.currentFrame[ray.index].g, color.g);
		atomicAdd(&scene.currentFrame[ray.index].b, color.b);
		atomicAdd(&scene.currentFrame[ray.index].a, float(new_frame));
	}
}

__global__ void LAUNCH_BOUNDS connect(ShadowRay* sRays, SceneData scene, int rayBufferSize)
{
	while (true)
	{
		const int index = atomicAdd(&ray_nr_connect, 1);
		if (index >= shadow_ray_cnt) return;

		const ShadowRay& ray = sRays[index];
		if (MBVHNode::traverseMBVHShadow(ray.origin, ray.direction, ray.t, scene))
		{
			atomicAdd(&scene.currentFrame[ray.index].r, ray.color.r);
			atomicAdd(&scene.currentFrame[ray.index].g, ray.color.g);
			atomicAdd(&scene.currentFrame[ray.index].b, ray.color.b);
		}
	}
}

__global__ void draw_framebuffer(vec4* currentBuffer, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	const int index = x + y * width;
	const vec4 &color = currentBuffer[index];
	const vec3 col = vec3(color.r, color.g, color.b) / color.a;

	const vec3 exponent = vec3(1.0f / 2.2f);
	draw(x, y, vec4(glm::pow(col, exponent), 1.0f));
}

// Helper function for using CUDA to add vectors in parallel.
cudaError launchKernels(cudaArray_const_t array, Params& params, int samples, int rayBufferSize)
{
	static int frame = 1;
	cudaError err;

	err = cuda(BindSurfaceToArray(framebuffer, array));

	const auto* camera = params.camera;

	const vec3 w = camera->GetViewDirection();
	const vec3 up = camera->GetUp();
	const vec3 u = normalize(cross(w, up));
	const vec3 v = normalize(cross(u, w));

	vec3 hor, ver;

	if (params.width > params.height)
	{
		hor = u * camera->GetFOVDistance() * float(params.width) / float(params.height);
		ver = v * camera->GetFOVDistance();
	}
	else
	{
		hor = u * camera->GetFOVDistance();
		ver = v * camera->GetFOVDistance() * float(params.height) / float(params.width);
	}

	if (samples == 0)
		cuda(MemcpyToSymbol(primary_ray_cnt, &samples, sizeof(int)));

	generatePrimaryRays << <params.smCores * 8, 128 >> > (params.gpuRays, camera->GetPosition(), w, hor, ver, params.width, params.height,
		1.0f / float(params.width), 1.0f / float(params.height), rayBufferSize, frame);
	setGlobals << <1, 1 >> > (rayBufferSize, params.width, params.height);
	extend << <params.smCores * 8, 128 >> > (params.gpuRays, params.gpuScene, rayBufferSize);
	//shade << <params.smCores * 8, 128 >> > (params.gpuRays, params.gpuNextRays, params.gpuShadowRays, params.gpuScene, frame, rayBufferSize);
	shade_invalid << <params.smCores * 8, 64 >> > (params.gpuRays, params.gpuNextRays, params.gpuShadowRays, params.gpuScene, frame, rayBufferSize);
	//shade_lights << <params.smCores * 8, 128 >> > (params.gpuRays, params.gpuNextRays, params.gpuShadowRays, params.gpuScene, frame, rayBufferSize);
	shade_microfacet << <params.smCores * 8, 64 >> > (params.gpuRays, params.gpuNextRays, params.gpuShadowRays, params.gpuScene, frame, rayBufferSize);

	connect << <params.smCores * 8, 128 >> > (params.gpuShadowRays, params.gpuScene, rayBufferSize);

	dim3 dimBlock(16, 16);
	dim3 dimGrid((params.width + dimBlock.x - 1) / dimBlock.x, (params.height + dimBlock.y - 1) / dimBlock.y);
	draw_framebuffer << <dimGrid, dimBlock >> > (params.gpuScene.currentFrame, params.width, params.height);

	cuda(DeviceSynchronize());

	frame++;
	if (frame == INT_MAX) frame = 1;
	std::swap(params.gpuRays, params.gpuNextRays);
	return err;
}