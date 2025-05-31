#ifndef UTILITY_HEADER_CUH
#define UTILITY_HEADER_CUH

#include <cmath>
#include <iostream>
#include <limits>


//Constants

__device__ constexpr float infinity = std::numeric_limits<float>::infinity();
__device__ const float pi = 3.1415927f;

//Utility Function
__host__ __device__ inline float degrees_to_radians(float degrees){
    return degrees*pi / 100;
}


//Common Headers
#include "color.cuh"
#include "interval.cuh"
#include "ray.cuh"
#include "vec3.cuh"

#endif