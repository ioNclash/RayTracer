#include "color.cuh"
#include "ray.cuh"
#include "vec3.cuh"

#include <iostream>

__device__ color ray_color(const ray&r) {
    vec3 unit_direction = unit_vector(r.direction());
    float a = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-a)*color(1.0f, 1.0f, 1.0f) + a*color(0.5f, 0.7f, 1.0f);
}

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line){
    if(result){
        std::cerr << "CUDA error =" << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";

        //Ensure device reset on exit
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(color *fb, int max_x, int max_y,
    point3 bottom_left_pixel, vec3  pixel_width, vec3 pixel_height, point3 camera_center){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return; //Due to block size, prevent rendering outside of image

    point3 pixel_center = bottom_left_pixel + (i*pixel_width) + (j*pixel_height);
    vec3 ray_direction = pixel_center - camera_center;
    ray r(camera_center,ray_direction);
    int pixel_index = j*max_x + i; //get index in frame buffer
    fb[pixel_index] = ray_color(r);
}


__host__ int main(){

    //Image
    float aspect_ratio= 16.0f/9.0f;
    int image_width = 1600;  // image width
    int image_height = int(image_width/aspect_ratio);
    image_height = (image_height <1 ) ? 1 : image_height;   // image height

    //Camera
    float focal_length = 1.0f;
    float viewport_height = 2.0f;
    float viewport_width = viewport_height * (float(image_width)/float(image_height));
    point3 camera_center = point3(0,0,0);

    //Calculate vectors across viewport edges
    vec3 viewport_u = vec3(viewport_width,0,0);
    vec3 viewport_v = vec3(0,viewport_height,0); //+ here to make pixel00 the bottom left pixel

    vec3 pixel_delta_u = viewport_u/image_width;
    vec3 pixel_delta_v = viewport_v/image_height;

    point3 viewport_bottom_left = camera_center - vec3(0,0,focal_length) - viewport_u/2 - viewport_v/2;
    point3 pixel00_loc = viewport_bottom_left + 0.5f * (pixel_delta_u + pixel_delta_v);


    //CUDA Image Division
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = image_width*image_height;
    size_t fb_size = 3*num_pixels*sizeof(float); //frame buffer

    //Allocate frame buffer
    color *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb , fb_size));


    //Render buffer

    dim3 blocks(image_width/tx+1,image_height/ty+1);
    dim3 threads(tx,ty);
    render<<<blocks, threads>>>(fb,image_width,image_height,pixel00_loc,pixel_delta_u,pixel_delta_v,camera_center);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //Write image
    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = image_height-1; j >= 0; j--) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j*image_width + i;
            write_color(std::cout,fb[pixel_index]);
        }
    }
    checkCudaErrors(cudaFree(fb));
}



