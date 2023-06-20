#pragma once

#include "cstdint"
#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <vector>

template <typename S>
void mandelbrot(std::vector<int32_t, S> img, uint32_t xdim, uint32_t ydim);
