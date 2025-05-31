#include "utility_header.cuh"

#include "camera.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"


__global__ void create_world(hittable **d_list, hittable **d_world ){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5f);
        *(d_list+1) = new sphere(vec3(0,-100.5f,-1), 100);
        *d_world    = new hittable_list(d_list,2);
    }

}

__global__ void free_world(hittable **d_list, hittable **d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}


__host__ int main(){
    //Create camera
    camera cam;

    cam.set_aspect_ratio(16.0f/9.0f);
    cam.set_image_width(1600);

    //Allocate Camera Data
    camera_data *d_cam_data;
    checkCudaErrors(cudaMallocManaged((void**)&d_cam_data, sizeof(camera_data)));
    *d_cam_data = cam.get_camera_data();

    //CUDA Image Division
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << d_cam_data->image_width << "x" << d_cam_data->image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = d_cam_data->image_width*d_cam_data->image_height;
    size_t fb_size = 3*num_pixels*sizeof(float); //frame buffer

    //Allocate frame buffer
    color *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb , fb_size));



    //Make World of hittables
    hittable **d_list;
    checkCudaErrors(cudaMalloc((void **) &d_list, 2*sizeof(hittable *)));

    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **) &d_world, sizeof(hittable *)));
    create_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    //Render buffer

    dim3 blocks(d_cam_data->image_width/tx+1,d_cam_data->image_height/ty+1);
    dim3 threads(tx,ty);
    render<<<blocks, threads>>>(fb,d_world,d_cam_data);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //Write image
    std::cout << "P3\n" << d_cam_data->image_width << " " << d_cam_data->image_height << "\n255\n";
    for (int j = d_cam_data->image_height-1; j >= 0; j--) {
        for (int i = 0; i < d_cam_data->image_width; i++) {
            size_t pixel_index = j*d_cam_data->image_width + i;
            write_color(std::cout,fb[pixel_index]);
        }
    }

    //Cleanup
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_cam_data));

    
    cudaDeviceReset();
}



