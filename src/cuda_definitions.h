#pragma once

#ifndef __CUDACC__
#define __launch_bounds__(x, y)

int atomicAdd(void*, unsigned int) {};
int atomicAggInc(int *ctr) {};

template <typename T, int TT>
class surface {
};
template <typename T, int TT>
class texture {
};
#endif