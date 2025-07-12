# Ray Tracer 
![Ray Tracer](https://github.com/user-attachments/assets/5ae70196-4e20-4198-a159-33a33023292d)

A Path Tracer written in C++, based on the '[_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html)' series and then modified for parallelism.

## Table of Contents
- [Implementation](#implementation)
- [How to Use](#how-to-use)
- [Credits](#credits)

## Implementation
This project was written in C++ and the CUDA toolkit, with custom classes defining vectors, rays, colors, camera, etc. This model currently allows for diffuse, reflective, and dielectric materials and spherical hit volumes. During image rendering, the canvas is divided into segments, each handled by different threads for speed of processing. Output is in a .ppm format.

## How to Use
To use, just run the ```Easter@Egg:~$ console make ``` command in terminal ensuring you have nvcc and an NVIDIA GPU (sorry intel and amd users), and view the Output from the /Outputs folder, if you have eog installed run ```console Easter@Egg:~$ make show ``` in terminal to view on completion!
```

# Credits
Credit to Peter Shirley, Trevor D Black and Steve Hollasch the authors of the [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html) book series, and the team behind maintaining it






