#include "CUDA/Kernel.cuh"

#include <cuda_runtime.h>
#include <surface_functions.h>
#include <cuda_surface_types.h>
#include <device_launch_parameters.h>

#include "CUDA/CudaDefinitions.h"
#include "CUDA/CudaAssert.h"

#include "Core/SceneData.cuh"
#include "BVH/MBVHNode.cuh"
#include "Core/Random.cuh"
#include "Core/Ray.cuh"
#include "Core/Material.cuh"
#include "Core/Triangle.cuh"
#include "Core/BSDF.h"

using namespace glm;

#ifdef __CUDACC__
#define LAUNCH_BOUNDS __launch_bounds__(128, 4)
#else
#define LAUNCH_BOUNDS
#endif

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

//Number of shadow rays generated in shade step, which are placed in connect step.
__device__ int shadow_ray_cnt = 0;

surface<void, cudaSurfaceType2D> framebuffer;

__device__ inline void draw(unsigned int x, unsigned int y, const vec4& color)
{
	surf2Dwrite(color, framebuffer, x * sizeof(vec4), y);
}

__device__ inline void draw_unbounded(unsigned int x, unsigned int y, const vec4& color)
{
	surf2Dwrite(color, framebuffer, x * sizeof(vec4), y, cudaBoundaryModeZero);
}

__device__ inline float balancePDFs(float pdf1, float pdf2)
{
	const float sum = pdf1 + pdf2;
	const float w1 = pdf1 / sum;
	const float w2 = pdf2 / sum;
	return max(0.0f, 1.0f / (w1 * pdf1 + w2 * pdf2));
}

__device__ inline float balanceHeuristic(float nf, float fPdf, float ng, float gPdf)
{
	return max(0.0f, (nf * fPdf) / (nf * fPdf + ng * gPdf));
}

__device__ inline float powerHeuristic(float nf, float fPdf, float ng, float gPdf)
{
	const float f = nf * fPdf;
	const float g = ng * gPdf;
	return max(0.0f, (f * f) / (f * f + g * g));
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
}

__global__ void LAUNCH_BOUNDS shade(Ray* rays, Ray* eRays, ShadowRay* sRays, SceneData scene, unsigned int frame, int activePaths, int pathLength)
{
	const int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= activePaths) return;

	Ray& ray = rays[index];
	if (!ray.valid())
	{
		if (scene.skyboxEnabled)
		{
			const vec2 uv = { 1.0f + atan2f(ray.direction.x, -ray.direction.z) * glm::one_over_pi<float>() * 0.5f, 1.0f - acosf(ray.direction.y) * glm::one_over_pi<float>() };
			const vec3 color = ray.throughput / ray.lastBsdfPDF * vec3(scene.getTextureColor(scene.skyboxTexture, uv));
			scene.currentFrame[ray.index] += vec4(color, 1.0f);
		}
		return;
	}

	const uvec3 tIdx = scene.indices[ray.hit_idx];
	vec3 N = scene.centerNormals[ray.hit_idx];
	const vec3 bary = triangle::getBaryCoords(ray.origin, N, scene.vertices[tIdx.x], scene.vertices[tIdx.y], scene.vertices[tIdx.z]);
	vec3 iN = triangle::getNormal(bary, scene.normals[tIdx.x], scene.normals[tIdx.y], scene.normals[tIdx.z]);
	const vec2 tCoords = triangle::getTexCoords(bary, scene.texCoords[tIdx.x], scene.texCoords[tIdx.y], scene.texCoords[tIdx.z]);
	const Material& mat = scene.gpuMaterials[scene.gpuMatIdxs[ray.hit_idx]];
	if (mat.normalTex >= 0)
		iN = sampleToWorld(scene.getTextureNormal(mat.normalTex, tCoords), iN);

	if (mat.is_light())
	{
		if (!scene.indirect)
			return;

		const float DdotNL = -dot(ray.direction, iN);
		vec3 color;

		if (ray.lastBounceType == Ray::SPECULAR)
		{
			color = ray.throughput * mat.emission;
		}
		else
		{
			const float lightPDF = ray.t * ray.t / (DdotNL * triangle::getArea(scene.vertices[tIdx.x], scene.vertices[tIdx.y], scene.vertices[tIdx.z])) * scene.lightCount;
			color = ray.throughput * mat.albedo * (1.0f / (ray.lastBsdfPDF + lightPDF));
		}

		color = max(vec3(0), color);

		scene.currentFrame[ray.index] += vec4(color, 1);
		return;
	}

	ray.origin = ray.getHitpoint();
	ray.throughput = max(vec3(0), ray.throughput / ray.lastBsdfPDF);
	const float flip = (dot(ray.direction, N) > 0) ? -1.0f : 1.0f;
	N *= flip;					  // Fix geometric normal
	iN *= flip;					  // Fix interpolated normal (consistent normal interpolation)

	uint seed = WangHash(ray.index * 16789 + frame * 1791 + pathLength * 720898027);

	const vec3 matColor = mat.diffuseTex < 0 ? mat.albedo : scene.getTextureColor(mat.diffuseTex, tCoords);
	ShadingData data;
	vec3 T, B;
	convertToLocalSpace(iN, &T, &B);
	data.color = matColor;
	data.roughness = mat.roughness;
	data.eta = mat.refractIdx;

	vec3 wo;
	float pdf;
	BSDFType reflectedType;
	const vec3 BSDF = SampleBSDF(data, iN, N, T, B, -ray.direction, RandomFloat(seed), RandomFloat(seed), wo, pdf, reflectedType);
	ray.lastBounceType = reflectedType == DIFFUSE ? Ray::DIFFUSE : Ray::SPECULAR;
	ray.direction = wo;

	if (reflectedType == DIFFUSE && scene.shadow && scene.lightCount > 0)
	{
		const int light = RandomIntMax(seed, scene.lightCount - 1);
		const uvec3 lightIdx = scene.indices[scene.lightIndices[light]];
		const vec3 lightPos = triangle::getRandomPointOnSurface(scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z], RandomFloat(seed), RandomFloat(seed));
		vec3 L = lightPos - ray.origin;
		const float sqDist = dot(L, L);
		const float distance = sqrtf(sqDist);
		L /= distance;

		const vec3 lN = scene.centerNormals[scene.lightIndices[light]];
		const float NdotL = dot(iN, L);
		const float LNdotL = -dot(lN, L);

		if (NdotL > 0 && LNdotL > 0)
		{
			const float area = triangle::getArea(scene.vertices[lightIdx.x], scene.vertices[lightIdx.y], scene.vertices[lightIdx.z]);
			const auto emission = scene.gpuMaterials[scene.gpuMatIdxs[light]].emission;
			float shadowPDF;
			const vec3 sampledBSDF = EvaluateBSDF(data, iN, T, -ray.direction, L, shadowPDF);
			if (shadowPDF > 0)
			{
				const int shadowIdx = atomicAdd(&shadow_ray_cnt, 1);
				const float lightPDF = sqDist / (LNdotL * area) * scene.lightCount;
				const vec3 shadowCol = ray.throughput * emission * sampledBSDF * (NdotL / (shadowPDF + lightPDF));

				sRays[shadowIdx] = ShadowRay(
					ray.origin + scene.normalEpsilon * L, L, shadowCol,
					distance - 2.0f * scene.distEpsilon, ray.index
				);
			}
		}

		ray.lastBounceType = Ray::DIFFUSE;
	}

	if (pdf < 1e-6f || isnan(pdf) || any(isnan(ray.throughput)) || all(lessThanEqual(ray.throughput, vec3(0.0f))))
		return; // Early out in case we have an invalid bsdf

	ray.throughput = max(vec3(0), ray.throughput * BSDF * abs(dot(iN, ray.direction)));
	ray.origin += N * scene.normalEpsilon;
	ray.lastNormal = iN;
	ray.lastBsdfPDF = pdf;
	int primary_index = atomicAdd(&primary_ray_cnt, 1);
	eRays[primary_index] = ray;
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
	const int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= rayBufferSize) return;

	unsigned int seed = (index + frame * 147565741) * 720898027;

	const int x = (start_position + index) % w;
	const int y = ((start_position + index) / w) % h;

	const float px = float(x) + RandomFloat(seed) - 0.5f;
	const float py = float(y) + RandomFloat(seed) - 0.5f;

	rays[index] = Ray::generate(origin, viewDir, hor, ver, px, py, invw, invh, x + y * w);
}

__global__ void LAUNCH_BOUNDS extend(Ray* rays, SceneData scene, int rayBufferSize)
{
	const int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index >= rayBufferSize) return;

	auto& ray = rays[index];
	ray.t = MAX_DISTANCE;
	MBVHNode::traverseMBVH(ray.origin, ray.direction, &ray.t, &ray.hit_idx, scene);
}

__global__ void LAUNCH_BOUNDS connect(ShadowRay* sRays, SceneData scene, int rayBufferSize)
{
	const int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index >= shadow_ray_cnt) return;

	const ShadowRay& ray = sRays[index];
	if (!MBVHNode::traverseMBVHShadow(ray.origin, ray.direction, ray.t, scene))
		scene.currentFrame[ray.index] += vec4(ray.color, 1);
}

__global__ void draw_framebuffer(vec4* currentBuffer, int width, int height, float frame)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	const int index = x + y * width;
	const vec4& color = currentBuffer[index];
	const vec3 col = vec3(color.r, color.g, color.b) / frame;

	const vec3 exponent = vec3(1.0f / 2.2f);
	draw(x, y, vec4(glm::pow(col, exponent), 1.0f));
}

__host__ inline void sample(int& frame, Params& params, int rayBufferSize)
{
	const auto* camera = params.camera;

	const vec3 w = camera->getForward();
	const vec3 up = camera->getUp();
	const vec3 u = normalize(cross(w, up));
	const vec3 v = normalize(cross(u, w));

	vec3 hor, ver;

	const float dist = camera->getPlaneDistance();
	hor = u * dist;
	ver = v * dist;
	const float aspectRatio = float(params.width) / float(params.height);
	if (params.width > params.height)
		hor *= aspectRatio;
	else
		ver *= aspectRatio;

	setGlobals << <1, 1 >> > (rayBufferSize, params.width, params.height);

	int activePaths = params.width * params.height;
	int groups = activePaths / 128;
	int groupSize = 128;
	int connectCount = 0;

	generatePrimaryRays << <groups, groupSize >> > (params.gpuRays, camera->getPosition(), w, hor, ver, params.width, params.height,
		1.0f / float(params.width), 1.0f / float(params.height), activePaths, frame);
	cuda(DeviceSynchronize());

	extend << <groups, groupSize >> > (params.gpuRays, params.gpuScene, activePaths);
	cuda(DeviceSynchronize());

	shade << <groups, groupSize >> > (params.gpuRays, params.gpuNextRays, params.gpuShadowRays, params.gpuScene, frame, activePaths, 0);
	cuda(DeviceSynchronize());

	cudaMemcpyFromSymbol(&connectCount, shadow_ray_cnt, sizeof(uint), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&activePaths, primary_ray_cnt, sizeof(uint), 0, cudaMemcpyDeviceToHost);
	std::swap(params.gpuRays, params.gpuNextRays);
	if (connectCount > 0 && params.gpuScene.shadow)
	{
		const int shadowGroups = (connectCount + (connectCount % 128)) / 128;
		const int shadowSize = 128;

		connect << <shadowGroups, shadowSize >> > (params.gpuShadowRays, params.gpuScene, connectCount);
	}

	constexpr int maxPathLength = 5;
	for (int i = 1; i < maxPathLength; i++)
	{
		setGlobals << <1, 1 >> > (rayBufferSize, params.width, params.height);
		activePaths = activePaths + (activePaths % 128);

		groups = activePaths / 128;
		groupSize = 128;
		extend << <groups, groupSize >> > (params.gpuRays, params.gpuScene, activePaths);
		cuda(DeviceSynchronize());
		shade << <groups, groupSize >> > (params.gpuRays, params.gpuNextRays, params.gpuShadowRays, params.gpuScene, frame, activePaths, i + 1);
		cuda(DeviceSynchronize());
		cudaDeviceSynchronize();

		int connectCount;
		cudaMemcpyFromSymbol(&connectCount, shadow_ray_cnt, sizeof(uint), 0, cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbol(&activePaths, primary_ray_cnt, sizeof(uint), 0, cudaMemcpyDeviceToHost);

		if (connectCount > 0 && params.gpuScene.shadow)
		{
			const int shadowGroups = (connectCount + (connectCount % 128)) / 128;
			const int shadowSize = 128;

			connect << <shadowGroups, shadowSize >> > (params.gpuShadowRays, params.gpuScene, connectCount);
			cuda(DeviceSynchronize());
		}

		std::swap(params.gpuRays, params.gpuNextRays);
	}
	frame++;
	cuda(DeviceSynchronize());
	return;

}

cudaError launchKernels(cudaArray_const_t array, Params& params, int rayBufferSize)
{
	cudaError err;

	err = cuda(BindSurfaceToArray(framebuffer, array));
	if (params.samples == 0)
		cuda(MemcpyToSymbol(primary_ray_cnt, &params.samples, sizeof(int)));

	sample(params.samples, params, rayBufferSize);
	params.samples++;

	dim3 dimBlock(16, 16);
	dim3 dimGrid((params.width + dimBlock.x - 1) / dimBlock.x, (params.height + dimBlock.y - 1) / dimBlock.y);
	draw_framebuffer << <dimGrid, dimBlock >> > (params.gpuScene.currentFrame, params.width, params.height, params.samples);

	cuda(DeviceSynchronize());
	return err;
}