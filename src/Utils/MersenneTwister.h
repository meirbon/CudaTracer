#include "Utils/RandomGenerator.h"

class MersenneTwister : public RandomGenerator
{
  public:
    MersenneTwister()
    {
        std::random_device mt_rd;
        mt_gen = std::mt19937(mt_rd()); // seed rng
    }

    float Rand(float range) override
    {
        return RandomUint() * 2.3283064365387e-10f * range;
    }

    unsigned int RandomUint() override { return mt_gen(); }

  private:
    std::mt19937 mt_gen;
};