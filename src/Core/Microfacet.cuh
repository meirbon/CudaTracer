#pragma once

#include <cuda_runtime.h>

#define GLM_FORCE_PURE
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/constants.hpp>

using namespace glm;

namespace microfacet
{
	// BeckmannDistribution Public Methods
	__device__ __host__ inline static float RoughnessToAlpha(float roughness)
	{
		roughness = glm::max(roughness, 1e-3f);
		const float x = logf(roughness);
		return min(1.0f, (1.62142f + 0.819955f * x + 0.1734f * x * x +
			0.0171201f * x * x * x + 0.000640711f * x * x * x * x));
	}

	__device__ __host__ inline vec3 SphericalDirection(float sinTheta, float cosTheta, float phi)
	{
		return vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
	}

	__device__ __host__ inline vec3 SphericalDirection(float sinTheta, float cosTheta, float phi,
		const vec3 & x, const vec3 & y,
		const vec3 & z
	)
	{
		return sinTheta * cosf(phi)* x + sinTheta * sinf(phi) * y + cosTheta * z;
	}

	__device__ __host__ inline float ErfInv(float x) {
		float w, p;
		x = glm::clamp(x, -.999f, .999f);
		w = -logf((1.0f - x) * (1.0f + x));
		if (w < 5.0f)
		{
			w = w - 2.5f;
			p = 2.81022636e-08f;
			p = 3.43273939e-07f + p * w;
			p = -3.5233877e-06f + p * w;
			p = -4.39150654e-06f + p * w;
			p = 0.00021858087f + p * w;
			p = -0.00125372503f + p * w;
			p = -0.00417768164f + p * w;
			p = 0.246640727f + p * w;
			p = 1.50140941f + p * w;
		}
		else
		{
			w = sqrtf(w) - 3.f;
			p = -0.000200214257f;
			p = 0.000100950558f + p * w;
			p = 0.00134934322f + p * w;
			p = -0.00367342844f + p * w;
			p = 0.00573950773f + p * w;
			p = -0.0076224613f + p * w;
			p = 0.00943887047f + p * w;
			p = 1.00167406f + p * w;
			p = 2.83297682f + p * w;
		}
		return p * x;
	}

	__device__ __host__ inline float Erf(float x)
	{
		// Save the sign of x
		int sign = 1;
		if (x < 0.0f) sign = -1;
		x = fabs(x);

		// A&S formula 7.1.26
		const float t = 1.0f / (1.0f + 0.3275911f * x);
		const float y = 1.0f -
			(((((1.061405429f * t + -1.453152027f) * t) + 1.421413741f) * t + -0.284496736f) * t + 0.254829592f) * t * std::exp(-x * x);

		return sign * y;
	}

	__device__ __host__ inline float CosTheta(const vec3 & w) { return w.z; }

	__device__ __host__ inline float Cos2Theta(const vec3 & w) { return w.z* w.z; }

	__device__ __host__ inline float AbsCosTheta(const vec3 & w) { return fabs(w.z); }

	__device__ __host__ inline float Sin2Theta(const vec3 & w)
	{
		return glm::max(0.f, 1.f - Cos2Theta(w));
	}

	__device__ __host__ inline float SinTheta(const vec3 & w) { return std::sqrt(Sin2Theta(w)); }

	__device__ __host__ inline float TanTheta(const vec3 & w) { return SinTheta(w) / CosTheta(w); }

	__device__ __host__ inline float Tan2Theta(const vec3 & w) {
		return Sin2Theta(w) / Cos2Theta(w);
	}

	__device__ __host__ inline float CosPhi(const vec3 & w) {
		float sinTheta = SinTheta(w);
		return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
	}

	__device__ __host__ inline float SinPhi(const vec3 & w) {
		float sinTheta = SinTheta(w);
		return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
	}

	__device__ __host__ inline float Cos2Phi(const vec3 & w) { return CosPhi(w)* CosPhi(w); }

	__device__ __host__ inline float Sin2Phi(const vec3 & w) { return SinPhi(w)* SinPhi(w); }

	__device__ __host__ inline float CosDPhi(const vec3 & wa, const vec3 & wb) {
		return glm::clamp(
			(wa.x * wb.x + wa.y * wb.y) / sqrtf((wa.x * wa.x + wa.y * wa.y) *
			(wb.x * wb.x + wb.y * wb.y)),
			-1.f, 1.f);
	}

	__device__ __host__ inline float G1(float lambda_w)
	{
		return 1.0f / (1.0f + lambda_w);
	}

	__device__ __host__ inline float D(const vec3 & wh, float alphay, float alphax)
	{
		const float tan2Theta = Tan2Theta(wh);
		if ((2.0f * tan2Theta) == tan2Theta) return 0.f;

		const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

		return expf(-tan2Theta * (Cos2Phi(wh) / (alphax * alphax) +
			Sin2Phi(wh) / (alphay * alphay))) /
			(glm::pi<float>() * alphax * alphay * cos4Theta);
	}

	__device__ __host__ inline void BeckmannSample11(float cosThetaI, float r1, float r2, float* slope_x, float* slope_y) {
		/* Special case (normal incidence) */
		if (cosThetaI > .9999f) {
			const float r = sqrtf(-std::log(1.0f - r1));
			const float sinPhi = sinf(2 * glm::pi<float>() * r2);
			const float cosPhi = cosf(2 * glm::pi<float>() * r2);
			*slope_x = r * cosPhi;
			*slope_y = r * sinPhi;
			return;
		}

		/* The original inversion routine from the paper contained
		   discontinuities, which causes issues for QMC integration
		   and techniques like Kelemen-style MLT. The following code
		   performs a numerical inversion with better behavior */
		const float sinThetaI = sqrtf(glm::max((float)0, (float)1 - cosThetaI * cosThetaI));
		const float tanThetaI = sinThetaI / cosThetaI;
		const float cotThetaI = 1 / tanThetaI;

		/* Search interval -- everything is parameterized
		   in the Erf() domain */
		float a = -1.0f;
		float c = Erf(cotThetaI);
		const float sample_x = glm::max(r1, (float)1e-6f);

		/* Start with a good initial guess */
		// float b = (1-sample_x) * a + sample_x * c;

		/* We can do better (inverse of an approximation computed in
		 * Mathematica) */
		const float thetaI = std::acos(cosThetaI);
		const float fit = 1.0f + thetaI * (-0.876f + thetaI * (0.4265f - 0.0594f * thetaI));
		float b = c - (1.0f + c) * powf(1.0f - sample_x, fit);

		/* Normalization factor for the CDF */
		const float normalization = 1.0f / (1.0f + c + 1.f / root_pi<float>() * tanThetaI * exp(-cotThetaI * cotThetaI));

		int it = 0;
		while (++it < 10) {
			/* Bisection criterion -- the oddly-looking
			   Boolean expression are intentional to check
			   for NaNs at little additional cost */
			if (!(b >= a && b <= c)) b = 0.5f * (a + c);

			/* Evaluate the CDF and its derivative
			   (i.e. the density function) */
			const float invErf = ErfInv(b);
			const float value = normalization *
				(1.0f + b + 1.f / sqrtf(glm::pi<float>()) * tanThetaI * std::exp(-invErf * invErf)) -
				sample_x;
			float derivative = normalization * (1.f - invErf * tanThetaI);

			if (std::abs(value) < 1e-5f) break;

			/* Update bisection intervals */
			if (value > 0)
				c = b;
			else
				a = b;

			b -= value / derivative;
		}

		/* Now convert back into a slope value */
		*slope_x = ErfInv(b);

		/* Simulate Y component */
		*slope_y = ErfInv(2.0f * glm::max(r2, (float)1e-6f) - 1.0f);
	}

	__device__ __host__ inline vec3 BeckmannSample(const vec3 & wi, float alpha_x, float alpha_y, float r1, float r2) {
		// 1. stretch wi
		vec3 wiStretched = normalize(vec3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

		// 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
		float slope_x, slope_y;
		BeckmannSample11(CosTheta(wiStretched), r1, r2, &slope_x, &slope_y);

		// 3. rotate
		float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
		slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
		slope_x = tmp;

		// 4. unstretch
		slope_x = alpha_x * slope_x;
		slope_y = alpha_y * slope_y;

		// 5. compute normal
		return normalize(vec3(-slope_x, -slope_y, 1.f));
	}

	__device__ __host__ inline void TrowbridgeReitzSample11(float cosTheta, float r1, float r2,
		float* slope_x, float* slope_y) {
		// special case (normal incidence)
		if (cosTheta > .9999f) {
			float r = sqrtf(r1 / (1 - r1));
			float phi = 6.28318530718f * r2;
			*slope_x = r * cos(phi);
			*slope_y = r * sin(phi);
			return;
		}

		const float sinTheta = sqrtf(max(0.f, 1.f - cosTheta * cosTheta));
		const float tanTheta = sinTheta / cosTheta;
		float a = 1.f / tanTheta;
		const float G1 = 2.f / (1.f + std::sqrt(1.f + 1.f / (a * a)));

		// sample slope_x
		const float A = 2.f * r1 / G1 - 1.f;
		float tmp = 1.f / (A * A - 1.f);
		if (tmp > 1e10) tmp = 1e10f;
		const float B = tanTheta;
		const float D = sqrtf(max(B * B * tmp * tmp - (A * A - B * B) * tmp, 0.0f));
		float slope_x_1 = B * tmp - D;
		float slope_x_2 = B * tmp + D;
		*slope_x = (A < 0.0f || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

		// sample slope_y
		float S;
		if (r2 > 0.5f) {
			S = 1.f;
			r2 = 2.f * (r2 - .5f);
		}
		else {
			S = -1.f;
			r2 = 2.f * (.5f - r2);
		}
		float z =
			(r2 * (r2 * (r2 * 0.27385f - 0.73369f) + 0.46341f)) /
			(r2 * (r2 * (r2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
		*slope_y = S * z * sqrtf(1.f + *slope_x * *slope_x);
	}

	__device__ __host__ inline vec3 TrowbridgeReitzSample(const vec3 & wi, float alpha_x, float alpha_y, float r1, float r2) {
		// 1. stretch wi
		const vec3 wiStretched = normalize(vec3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

		// 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
		float slope_x, slope_y;
		TrowbridgeReitzSample11(CosTheta(wiStretched), r1, r2, &slope_x, &slope_y);

		// 3. rotate
		const float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
		slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
		slope_x = tmp;

		// 4. unstretch
		slope_x = alpha_x * slope_x;
		slope_y = alpha_y * slope_y;

		// 5. compute normal
		return normalize(vec3(-slope_x, -slope_y, 1.f));
	}

	struct Microfacet
	{
		Microfacet(float alphax, float alphay, bool visibility = true)
		{
			this->alphaX = alphax;
			this->alphaY = alphay;
			this->sampleVisibility = visibility;
		}

		union {
			vec2 alpha;
			struct {
				float alphaX, alphaY;
			};
		};
		bool sampleVisibility;

		__device__ __host__ inline static void sampleGGX_P22_11(float cosThetaI, float* slopex, float* slopey, float r1, float r2)
		{
			// The special case where the ray comes from normal direction
			// The following sampling is equivalent to the sampling of
			// micro facet normals (not slopes) on isotropic rough surface
			if (cosThetaI > 0.9999f) {
				const float r = sqrtf(r1 / (1.0f - r1));
				const float sinPhi = std::sin(glm::two_pi<float>() * r2);
				const float cosPhi = std::cos(glm::two_pi<float>() * r2);
				*slopex = r * cosPhi;
				*slopey = r * sinPhi;
				return;
			}

			const float sinThetaI = sqrtf(max(0.0f, 1.0f - cosThetaI * cosThetaI));
			const float tanThetaI = sinThetaI / cosThetaI;
			const float a = 1.0f / tanThetaI;
			const float G1 = 2.0f / (1.0f + sqrtf(1.0f + 1.0f / (a * a)));

			// Sample slope x
			const float A = 2.0f * r1 / G1 - 1.0f;
			const float B = tanThetaI;
			const float tmp = min(1.0f / (A * A - 1.0f), 1.0e12f);

			const float D = sqrtf(B * B * tmp * tmp - (A * A - B * B) * tmp);
			const float slopex1 = B * tmp - D;
			const float slopex2 = B * tmp + D;
			*slopex = (A < 0.0f || slopex2 > 1.0f / tanThetaI) ? slopex1 : slopex2;

			// Sample slope y
			float S;
			if (r2 > 0.5f)
				S = 1.0f, r2 = 2.0f * (r2 - 0.5f);
			else
				S = -1.0f, r2 = 2.0f * (0.5f - r2);

			const float z = (r2 * (r2 * (r2 * 0.27385f - 0.73369f) + 0.46341f)) /
				(r2 * (r2 * (r2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
			*slopey = S * z * sqrtf(1.0f + (*slopex) * (*slopex));
		}

		__device__ __host__ inline static vec3 sampleGGX(const vec3 & wi, float alphaX, float alphaY, float r1, float r2)
		{
			// 1. stretch wi
			const vec3 wiStretched = normalize(vec3(alphaX * wi.x, alphaY * wi.y, wi.z));

			// 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
			float slopex, slopey;
			sampleGGX_P22_11(CosTheta(wiStretched), &slopex, &slopey, r1, r2);

			// 3. rotate
			float  tmp = CosPhi(wiStretched) * slopex - SinPhi(wiStretched) * slopey;
			slopey = SinPhi(wiStretched) * slopex + CosPhi(wiStretched) * slopey;
			slopex = tmp;

			// 4. unstretch
			slopex = alphaX * slopex;
			slopey = alphaY * slopey;

			// 5. compute normal
			return normalize(vec3(-slopex, -slopey, 1.0f));
		}

		// GGX
		__device__ __host__ inline vec3 sample_ggx(const vec3 & wo, float r1, float r2) const
		{
			if (!sampleVisibility)
			{
				float tan2Theta, phi;
				if (alphaX == alphaY)
				{
					tan2Theta = alphaX * alphaX * r1 / (1.0f - r1);
					phi = 2.0f * glm::pi<float>() * r2;
				}
				else
				{
					phi = atanf(alphaY / alphaX * tanf(2.0f * glm::pi<float>() * r2 + 0.5f * glm::pi<float>()));
					if (r2 > 0.5f)
						phi += glm::pi<float>();

					const float sinPhi = sinf(phi);
					const float cosPhi = cosf(phi);
					const float alphaX2 = alphaX * alphaX;
					const float alphaY2 = alphaY * alphaY;
					const float alpha2 = 1.0f / (cosPhi * cosPhi / alphaX2 + sinPhi * sinPhi / alphaY2);
					tan2Theta = r1 / (1.0f - r1) * alpha2;
				}

				const float cosTheta = 1.0f / sqrtf(1.0f + tan2Theta);
				const float sinTheta = sqrtf(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
				const vec3 wm = { sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta };
				return wm.z < 0.0f ? -wm : wm;
			}
			else
			{
				bool flip = wo.z < 0.0f;
				const vec3 wm = sampleGGX(flip ? -wo : wo, alphaX, alphaY, r1, r2);

				if (wm.z < 0.0f)
					return -wm;

				return wm;
			}
		}

		__device__ __host__ inline float lambda_ggx(const glm::vec3 & wo) const
		{
			const float absTanThetaO = fabs(TanTheta(wo));
			if (2.0f * absTanThetaO == absTanThetaO)
				return 0.0f;

			const float alpha = sqrtf(Cos2Phi(wo) * alphaX * alphaX + Sin2Phi(wo) * alphaY * alphaY);
			const float alpha2Tan2Theta = alpha * absTanThetaO * alpha * absTanThetaO;
			return (-1.0f + sqrtf(1.0f + alpha2Tan2Theta)) / 2.0f;
		}

		__device__ __host__ inline float pdf_ggx(const vec3 & wo, const vec3 & wh, const vec3 & wi) const
		{
			if (!sampleVisibility)
			{
				const float temp = 1.0f / (1.0f + lambda_ggx(wo));
				const float temp2 = 1.0f / (1.0f + lambda_ggx(wi));
				return fabs(dot(wi, wh)) * temp * temp2 / max(1.0e-8f, fabs(CosTheta(wi) * CosTheta(wh)));
			}
			else
				return G1(lambda_ggx(wo));
		}

		// BECKMANN
		__device__ __host__ vec3 sample_beckmann(const glm::vec3 & wo, float r1, float r2) const
		{
			if (!sampleVisibility) {
				// Sample full distribution of normals for Beckmann distribution

				// Compute $\tan^2 \theta$ and $\phi$ for Beckmann distribution sample
				float tan2Theta, phi;
				if (alphaX == alphaY)
				{
					const float logSample = logf(1.0f - r1);
					tan2Theta = -alphaX * alphaX * logSample;
					phi = r2 * two_pi<float>();
				}
				else
				{
					// Compute _tan2Theta_ and _phi_ for anisotropic Beckmann
					// distribution
					const float logSample = logf(1 - r1);
					phi = atanf(alphaY / alphaX * tanf(2.0f * pi<float>() * r2 + 0.5f * pi<float>()));

					if (r2 > 0.5f) phi += pi<float>();
					const float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
					const float alphax2 = alphaX * alphaX;
					const float alphay2 = alphaY * alphaY;
					tan2Theta = -logSample / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
				}

				// Map sampled Beckmann angles to normal direction _wh_
				const float cosTheta = 1.f / sqrtf(1.f + tan2Theta);
				const float sinTheta = sqrtf(glm::max(0.f, 1.f - cosTheta * cosTheta));
				const vec3 wh = SphericalDirection(sinTheta, cosTheta, phi);

				if (wh.z < 0.f)
					return -wh;
				return wh;
			}
			else {
				// Sample visible area of normals for Beckmann distribution
				const bool flip = wo.z < 0.f;
				const vec3 wh = BeckmannSample(flip ? -wo : wo, alphaX, alphaY, r1, r2);

				if (flip)
					return -wh;
				return wh;
			}
		}

		__device__ __host__ inline float pdf_beckmann(const vec3 & wo, const vec3 & wh, const vec3 & wi) const
		{
			if (!sampleVisibility)
			{
				const float temp = 1.0f / (1.0f + lambda_beckmann(wo));
				const float temp2 = 1.0f / (1.0f + lambda_beckmann(wi));
				return fabs(dot(wi, wh)) * temp * temp2 / max(1.0e-8f, fabs(CosTheta(wi) * CosTheta(wh)));
			}
			else
				return G1(lambda_beckmann(wo));
		}

		__device__ __host__ inline float lambda_beckmann(const vec3 & w) const
		{
			const float absTanTheta = fabs(TanTheta(w));
			// Check for infinity
			if ((2.0f * absTanTheta) == absTanTheta) return 0.f;

			// Compute _alpha_ for direction _w_
			const float alpha = sqrtf(Cos2Phi(w) * alphaX * alphaX + Sin2Phi(w) * alphaY * alphaY);
			const float a = 1.0f / (alpha * absTanTheta);

			if (a >= 1.6f) return 0.f;

			return (1.f - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
		}

		__device__ __host__ vec3 sample_trowbridge_reitz(const vec3 & wo, float r1, float r2) const
		{
			float cosTheta = 0;
			float phi = glm::two_pi<float>() * r2;

			if (alphaX == alphaY)
			{
				float tanTheta2 = alphaX * alphaX * r1 / (1.0f - r1);
				cosTheta = 1.0f / std::sqrt(1.0f + tanTheta2);
			}
			else
			{
				phi = atanf(alphaY / alphaX * std::tan(2.0f * glm::pi<float>() * r2 + .5f * glm::pi<float>()));
				if (r2 > .5f) phi += glm::pi<float>();
				const float sinPhi = sinf(phi), cosPhi = std::cos(phi);
				const float alphax2 = alphaX * alphaX, alphay2 = alphaY * alphaY;
				const float alpha2 = 1.0f / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
				const float tanTheta2 = alpha2 * r1 / (1.0f - r1);
				cosTheta = 1.0f / sqrtf(1.0f + tanTheta2);
			}
			const float sinTheta = sqrtf(glm::max(0.f, 1.f - cosTheta * cosTheta));
			const vec3 wh = SphericalDirection(sinTheta, cosTheta, phi);
			if (wh.z < 0.0f)
				return -wh;
			return wh;
		}

		__device__ __host__ inline float pdf_trowbridge_reitz(const vec3 & wo, const vec3 & wh, const vec3 & wi) const
		{
			if (!sampleVisibility)
			{
				const float temp = 1.0f / (1.0f + lambda_trowbridge_reitz(wo));
				const float temp2 = 1.0f / (1.0f + lambda_trowbridge_reitz(wi));
				return fabs(dot(wi, wh)) * temp * temp2 / max(1.0e-8f, fabs(CosTheta(wi) * CosTheta(wh)));
			}
			else
				return G1(lambda_trowbridge_reitz(wo));
		}

		__device__ __host__ inline float lambda_trowbridge_reitz(const vec3 & w) const
		{
			const float absTanTheta = fabs(TanTheta(w));

			if ((2.0f * absTanTheta) == absTanTheta) return 0.f;

			// Compute _alpha_ for direction _w_
			const float alpha = sqrtf(Cos2Phi(w) * alphaX * alphaX + Sin2Phi(w) * alphaY * alphaY);
			const float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
			return (-1.f + sqrtf(1.f + alpha2Tan2Theta)) / 2.f;
		}
	};
}