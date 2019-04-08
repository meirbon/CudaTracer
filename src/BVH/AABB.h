#pragma once

#include <glm/glm.hpp>
#include <immintrin.h>

using namespace glm;

class AABB {
public:
	AABB() {
		for (int i = 0; i < 4; i++) {
			this->bmax[i] = this->bmin[i] = 0.0f;
		}
	};

	AABB(__m128 min, __m128 max) {
		bmin4 = min, bmax4 = max;
		bmin[3] = bmax[3] = 0;
	}

	AABB(glm::vec3 min, glm::vec3 max) {
		bmin[0] = min.x;
		bmin[1] = min.y;
		bmin[2] = min.z;
		bmin[3] = 0;

		bmax[0] = max.x;
		bmax[1] = max.y;
		bmax[2] = max.z;
		bmax[3] = 0;
	}

	__inline void Reset() { bmin4 = _mm_set_ps1(1e34f), bmax4 = _mm_set_ps1(-1e34f); }

	bool Contains(const __m128 &p) const {
		union {
			__m128 va4;
			float va[4];
		};
		union {
			__m128 vb4;
			float vb[4];
		};
		va4 = _mm_sub_ps(p, bmin4), vb4 = _mm_sub_ps(bmax4, p);
		return ((va[0] >= 0) && (va[1] >= 0) && (va[2] >= 0) && (vb[0] >= 0) && (vb[1] >= 0) && (vb[2] >= 0));
	}

	__inline void GrowSafe(const AABB &bb) {
		xMin = glm::min(xMin, bb.xMin);
		yMin = glm::min(yMin, bb.yMin);
		zMin = glm::min(zMin, bb.zMin);

		xMax = glm::max(xMax, bb.xMax);
		yMax = glm::max(yMax, bb.yMax);
		zMax = glm::max(zMax, bb.zMax);
	}

	__inline void Grow(const AABB &bb) {
		bmin4 = _mm_min_ps(bmin4, bb.bmin4);
		bmax4 = _mm_max_ps(bmax4, bb.bmax4);
	}

	__inline void Grow(const __m128 &p) {
		bmin4 = _mm_min_ps(bmin4, p);
		bmax4 = _mm_max_ps(bmax4, p);
	}

	__inline void Grow(const __m128 min4, const __m128 max4) {
		bmin4 = _mm_min_ps(bmin4, min4);
		bmax4 = _mm_max_ps(bmax4, max4);
	}

	__inline void Grow(const glm::vec3 &p) {
		__m128 p4 = _mm_setr_ps(p.x, p.y, p.z, 0);
		Grow(p4);
	}

	AABB Union(const AABB &bb) const {
		AABB r;
		r.bmin4 = _mm_min_ps(bmin4, bb.bmin4);
		r.bmax4 = _mm_max_ps(bmax4, bb.bmax4);
		return r;
	}

	static AABB Union(const AABB &a, const AABB &b) {
		AABB r;
		r.bmin4 = _mm_min_ps(a.bmin4, b.bmin4);
		r.bmax4 = _mm_max_ps(a.bmax4, b.bmax4);
		return r;
	}

	AABB Intersection(const AABB &bb) const {
		AABB r;
		r.bmin4 = _mm_max_ps(bmin4, bb.bmin4);
		r.bmax4 = _mm_min_ps(bmax4, bb.bmax4);
		return r;
	}

	__inline float Extend(const int axis) const { return bmax[axis] - bmin[axis]; }

	__inline float Minimum(const int axis) const { return bmin[axis]; }

	__inline float Maximum(const int axis) const { return bmax[axis]; }

	__inline float Volume() const {
		union {
			__m128 length4;
			float length[4];
		};
		length4 = _mm_sub_ps(this->bmax4, this->bmin4);
		return length[0] * length[1] * length[2];
	}

	__inline glm::vec3 Centroid() const {
		union {
			__m128 center;
			float c4[4];
		};
		center = Center();
		return glm::vec3(c4[0], c4[1], c4[2]);
	}

	__inline float Area() const {
		union {
			__m128 e4;
			float e[4];
		};
		e4 = _mm_sub_ps(bmax4, bmin4);
		return fmax(0.0f, e[0] * e[1] + e[0] * e[2] + e[1] * e[2]);
	}

	__inline glm::vec3 Lengths() const {
		//float length_x = bmax[0] - bmin[0];
		//float length_y = bmax[1] - bmin[1];
		//float length_z = bmax[2] - bmin[2];
		//return glm::vec3(length_x, length_y, length_z);
		union {
			__m128 length4;
			float length[4];
		};
		length4 = _mm_sub_ps(this->bmax4, this->bmin4);
		return glm::vec3(length[0], length[1], length[2]);
	}

	int LongestAxis() const {
		int a = 0;
		if (Extend(1) > Extend(0))
			a = 1;
		if (Extend(2) > Extend(a))
			a = 2;
		return a;
	}

	// data members
	union {
		struct {
			union {
				__m128 bmin4;
				float bmin[4];
				struct {
					float xMin, yMin, zMin;
					int leftFirst;
				};
			};
			union {
				__m128 bmax4;
				float bmax[4];
				struct {
					float xMax, yMax, zMax;
					int count;
				};
			};
		};
		__m128 bounds[2] = { _mm_set_ps(1e34f, 1e34f, 1e34f, 0), _mm_set_ps(-1e34f, -1e34f, -1e34f, 0) };
	};

	__inline void SetBounds(const __m128 min4, const __m128 max4) {
		bmin4 = min4;
		bmax4 = max4;
	}

	__inline __m128 Center() const {
		return _mm_mul_ps(_mm_add_ps(bmin4, bmax4), _mm_set_ps1(0.5f));
	}

	__inline float Center(uint axis) const { return (bmin[axis] + bmax[axis]) * 0.5f; }
};