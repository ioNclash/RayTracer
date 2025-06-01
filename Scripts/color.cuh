#ifndef COLOR_CUH
#define COLOR_CUH

#include "vec3.cuh"
#include "interval.cuh"

using color = vec3;

__host__ void write_color(std::ostream& out, const color& pixel_color){
    float r = pixel_color.x();
    float g = pixel_color.y();
    float b = pixel_color.z();

    // Write the translated [0,255] value of each color component.

    static const interval intensity(0.000f, 0.999f);
    int rbyte = int(256 * intensity.clamp(r));
    int gbyte = int(256 * intensity.clamp(g));
    int bbyte = int(256 * intensity.clamp(b));
    std::cout << rbyte << " " << gbyte << " " << bbyte << "\n";
}

#endif