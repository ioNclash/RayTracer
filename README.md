# Ray Tracer 
![Ray Tracer](https://github.com/user-attachments/assets/5ae70196-4e20-4198-a159-33a33023292d)

A Path Tracer written in C++, based on the '[_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html)' series and then modified for parallelism.

## Table of Contents
- [Implementation](#implementation)
- [How to Use](#how-to-use)
- [Credits](#credits)

## Implementation
This project was written in C++ using only the standard libraries, with custom classes defining vectors, rays, colors, camera, etc. This model currently allows for diffuse, reflective, and dielectric materials and spherical hit volumes. During image rendering, the canvas is divided into segments, each handled by different threads for speed of processing. Output is in a .ppm format.

## How to Use
To use, you can edit main.cpp to change the scene setup, altering the objects, image resolution and samples per pixel and camera position, focal distance and angle
Then build main.cpp, and run the executable, piping the output to a .ppm file, then  use a ppm viewer to see image. I use feh on linux, there is also the option of converting the ppm to png    
  
**Example of use in bash**

```console
nicholas@easteregg:~Documents/RayTracer$ g++ Scripts/main.cpp -o Builds/build
nicholas@easteregg:~Documents/RayTracer$ Builds/build > Outputs/image.ppm
nicholas@easteregg:~Documents/RayTracer$ feh Outputs/image.ppm
```

# Credits
Credit to Peter Shirley, Trevor D Black and Steve Hollasch the authors of the [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html) book series, and the team behind maintaining it






