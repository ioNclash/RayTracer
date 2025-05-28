#ifndef COLOR_CUH
#define COLOR_CUH

#include "vec3.cuh"
#include <iostream>

using color = vec3;

__host__ void write_color(std::ostream& out, const color& pixel_color){
    float r = pixel_color.x();
    float g = pixel_color.y();
    float b = pixel_color.z();

    int rbyte = int(255.99f*r);
    int gbyte = int(255.99f*g);
    int bbyte = int(255.99f*b);
    std::cout << rbyte << " " << gbyte << " " << bbyte << "\n";
}

#endif