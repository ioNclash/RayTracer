#include "utility_header.cuh"

#include "bvh.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "sphere.cuh"

__global__ void construct_hittable_list(hittable_list *world, hittable **buffer, int capacity) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        new(world) hittable_list(buffer, capacity);
}
__global__ void create_world(curandState *rand_state,hittable_list *world)
{
    curandState local_rand_state = *rand_state;

    // Add ground
    material *ground_material = new lambertian(color(0.5f, 0.5f, 0.5f));
    world->add(new sphere(vec3(0, -1000.0f, -1), 1000, ground_material));

    // Add random spheres
    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            float choose_mat = random_float(&local_rand_state);
            point3 center(a + 0.9f * random_float(&local_rand_state),
                          0.2f,
                          b + 0.9f * random_float(&local_rand_state));

            if ((center - point3(4, 0.2f, 0)).length() > 0.9f)
            {
                if (choose_mat < 0.8f)
                {
                    // Lambertian
                    color albedo = color::random(&local_rand_state) * color::random(&local_rand_state);
                    point3 center2 = center + vec3(0, random_float(&local_rand_state, 0, 0.5f), 0);
                    world->add(new sphere(center, center2, 0.2F, new lambertian(albedo)));
                }
                else if (choose_mat < 0.95f)
                {
                    // Metal
                    color albedo = color::random(&local_rand_state, 0.5f, 1.0f);
                    float fuzz = random_float(&local_rand_state) * 0.5f;
                    world->add(new sphere(center, 0.2f, new metal(albedo, fuzz)));
                }
                else
                {
                    // Dielectric
                    world->add(new sphere(center, 0.2, new dielectric(1.5)));
                }
            }
        }
    }

    // Add three big spheres
    world->add(new sphere(vec3(0, 1, 0), 1.0f, new dielectric(1.5)));
    world->add(new sphere(vec3(-4, 1, 0), 1.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f))));
    world->add(new sphere(vec3(4, 1, 0), 1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f)));

    *rand_state = local_rand_state;
}

__global__ void free_world(hittable_list *world)
{
    // Free all objects in the list
    for (int i = 0; i < world->get_size(); i++)
    {
        hittable *obj = world->get_item(i);
        if (obj)
        {
            sphere *s = (sphere *)obj;
            cudaFree( s->mat_ptr);
            cudaFree(s);
        }
    }
    // Free the world object itself
    delete world;
}

__host__ int main()
{
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Create camera
    camera cam;
    initialize_camera(
        cam,
        16.0f / 9.0f,     // Aspect ratio
        1200,             // Image width
        100,              // Samples per pixel
        50,               // Max depth
        20.0f,            // Field of view in degrees
        point3(13, 2, 3), // Look from

        point3(0, 0, 0), // Look at
        vec3(0, 1, 0),   // Up vector
        0.0f,            // Defocus angle
        10.0f            // Focus distance
    );

    // Allocate Camera Data
    camera *d_cam;
    checkCudaErrors(cudaMalloc((void **)&d_cam, sizeof(camera)));
    checkCudaErrors(cudaMemcpy(d_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice));

    // CUDA Image Division
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << cam.image_width << "x" << cam.image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = cam.image_width * cam.image_height;
    size_t fb_size = 3 * num_pixels * sizeof(float); // frame buffer

    // Allocate frame buffer
    color *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Set up CUDA Random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState))); // Random state for each pixel

    // Initialise Render

    dim3 blocks(cam.image_width / tx + 1, cam.image_height / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(d_rand_state, d_cam); // Initialize the random state and camera
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Make World of hittables
    int max_hittables = 22 * 22 + 1 + 3; // 22x22 spheres + 4 other spheres
    hittable **d_buffer;                 // Buffer for hittables
    checkCudaErrors(cudaMalloc((void **)&d_buffer, max_hittables * sizeof(hittable *)));

    hittable_list *d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable_list)));

    construct_hittable_list<<<1,1>>>(d_world, d_buffer, max_hittables);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();

    create_world<<<1, 1>>>(d_rand_state, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(d_rand_state, fb, (hittable*)d_world, d_cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Write image
    std::cout << "P3\n"
              << cam.image_width << " " << cam.image_height << "\n255\n";
    for (int j = cam.image_height - 1; j >= 0; j--)
    {
        for (int i = 0; i < cam.image_width; i++)
        {
            size_t pixel_index = j * cam.image_width + i;
            write_color(std::cout, fb[pixel_index]);
        }
    }

    // Cleanup
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_buffer));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_cam));

    cudaDeviceReset();
}
