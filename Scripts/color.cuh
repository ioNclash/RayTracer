#ifndef COLOR_CUH
#define COLOR_CUH

#include "vec3.cuh"
#include "interval.cuh"

using color = vec3;

inline __host__ float linear_to_gamma(float linear_component){
    if (linear_component > 0)
        return std::sqrt(linear_component);
    return 0;
}

__host__ void write_color(std::ostream& out, const color& pixel_color){
    float r = pixel_color.x();
    float g = pixel_color.y();
    float b = pixel_color.z();

    //Apply a linear to gamma correction to each color component.
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    // Write the translated [0,255] value of each color component.

    static const interval intensity(0.000f, 0.999f);
    int rbyte = int(256 * intensity.clamp(r));
    int gbyte = int(256 * intensity.clamp(g));
    int bbyte = int(256 * intensity.clamp(b));
    std::cout << rbyte << " " << gbyte << " " << bbyte << "\n";
}

#endif