#pragma once

namespace memory
{
#if defined(_MSC_VER)
#define ALIGN(x) __declspec(align(x))
#define MALLOC64(x) __aligned_malloc(x, 64)
#define FREE64(x) _aligned_free(x)
#elif defined(__APPLE__)
#define ALIGN(x) __attribute__((aligned(x)))
inline void *MALLOC64(size_t x)
{
    void *pointer;
    posix_memalign(&pointer, 64, x);
    return pointer;
}
#define FREE64(x) free(x)
#elif defined(__linux__)
#define ALIGN(x) __attribute__((aligned(x)))
#define MALLOC64(x) aligned_alloc(64, x)
#define FREE64(x) free(x)
#endif
} // namespace memory