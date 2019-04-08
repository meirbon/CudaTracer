#pragma once

#include <chrono>

namespace utils
{
struct Timer
{
	typedef std::chrono::high_resolution_clock Clock;
	typedef Clock::time_point TimePoint;
	typedef std::chrono::microseconds MicroSeconds;

	TimePoint start;

	inline Timer() : start(get()) {}

	inline float elapsed() const
	{
		auto diff = get() - start;
		auto duration_us = std::chrono::duration_cast<MicroSeconds>(diff);
		return static_cast<float>(duration_us.count()) / 1000.0f;
	}

	static inline TimePoint get() { return Clock::now(); }

	inline void reset() { start = get(); }
};
} // namespace utils
