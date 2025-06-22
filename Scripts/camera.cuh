#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "hittable.cuh"
#include "material.cuh"

struct camera{ //Trivial struct for CUDA kernel
    float aspect_ratio; // Aspect ratio of the image (width/height)
    int image_width; // Width of the image in pixels
    int image_height; // Height of the image in pixels, calculated from aspect ratio
    int samples_per_pixel; // Number of samples per pixel for anti-aliasing
    float pixel_sample_scale; // Scale factor for pixel color accumulation
    int max_depth; // Maximum iterative depth for ray tracing

    point3 center; // Center of the camera viewport
    point3 pixel00_loc; // Location of the bottom-left pixel in the viewport
    vec3 pixel_delta_u; // Vector representing the change in x for each pixel
    vec3 pixel_delta_v; // Vector representing the change in y for each pixel

    point3 look_from ; //Camera position
    point3 look_at; //Point the camera is looking at
    float fov ; //Field of view in degrees,
    vec3 vup; //Up direction for the camera
    vec3 u,v,w;

    float defocus_angle; //Variation angle of rays through each pixel
    float focus_distance; //Distance from camera look_from point to focus plane
    vec3 defocus_disk_u; //Defocus disk horizontal radius
    vec3 defocus_disk_v; //Defocus disk vertical radius

};


__host__ void initialize_camera(
    camera& cam,
    float aspect_ratio = 16.0f/9.0f,
    int image_width = 1200,
    int samples_per_pixel = 100,
    int max_depth = 50,
    float fov = 20.0f,

    point3 look_from = point3(13,2,3),
    point3 look_at = point3(0,0,0),
    vec3 vup = vec3(0,1,0),
    
    float defocus_angle = 0.0f,
    float focus_distance = 10.0f)
     {
    
        //Initialize camera parameters
        cam.aspect_ratio = aspect_ratio; // Set the aspect ratio of the camera
        cam.image_width = image_width; // Set the width of the image in pixels
        cam.samples_per_pixel = samples_per_pixel; // Set the number of samples per pixel for anti-aliasing
        cam.max_depth = max_depth; // Set the maximum depth for ray tracing
        cam.fov = fov; // Set the field of view in degrees
        cam.look_from = look_from; // Set the camera position
        cam.look_at = look_at; // Set the point the camera is looking at
        cam.vup = vup, // Set the up direction for the camera
        cam.defocus_angle = defocus_angle; // Set the defocus angle for depth of field
        cam.focus_distance = focus_distance;// Set the distance from the camera to the focus plane

      //Image
        cam.image_height = int(image_width/aspect_ratio);
        cam.image_height = (cam.image_height <1 ) ? 1 : cam.image_height;   // image height

        cam.pixel_sample_scale = 1.0f / float(samples_per_pixel);// Scale factor for pixel color accumulation

        cam.center = cam.look_from; // Set the camera center to the lookfrom position

        //Determine viewport dimensions
        float theta = degrees_to_radians(fov); // Convert field of view from degrees to radians
        float h = std::tan(theta/2);
        float viewport_height = 2.0f * h * focus_distance; // Height of the viewport
        float viewport_width = viewport_height * (float(cam.image_width)/float(cam.image_height)); //Viewport width based on aspect ratio
        
        //Calculate camera coordinate system
        cam.w = unit_vector(look_from - look_at); // Direction from camera to lookat point
        cam.u = unit_vector(cross(vup, cam.w)); // Perpendicular vector in the x direction
        cam.v = cross(cam.w, cam.u); // Perpendicular vector in the y direction

        //Calculate vectors across viewport edges
        vec3 viewport_u = viewport_width * cam.u; // Width vector of the viewport
        vec3 viewport_v = viewport_height * cam.v; // Height vector of the viewport

        //Calculate pixel delta
        cam.pixel_delta_u = viewport_u/cam.image_width;
        cam.pixel_delta_v = viewport_v/cam.image_height;

        //Calculate location of bottom left pixel
        point3 viewport_bottom_left = cam.center - (focus_distance*cam.w) - viewport_u/2 - viewport_v/2;
        cam.pixel00_loc = viewport_bottom_left + 0.5f * (cam.pixel_delta_u + cam.pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        float defocus_radius = focus_distance * std::tan(degrees_to_radians(defocus_angle / 2));
        cam.defocus_disk_u = cam.u * defocus_radius;
        cam.defocus_disk_v = cam.v * defocus_radius;
    }


__device__ color ray_color(curandState *rand_state,camera* cam, ray&r, hittable **world) { 
    ray current_ray = r;
    color current_attenuation = vec3(1.0f,1.0f,1.0f);
    for(int i=0; i<cam->max_depth; i++){
        hit_record rec;
        if ((*world)->hit(current_ray,interval(0.001f,infinity),rec)){
            ray scattered;
            color attenuation;
            if (rec.mat_ptr->scatter(rand_state,current_ray,rec,attenuation,scattered)){
                current_attenuation *= attenuation;
                current_ray = scattered;
            }
            else{
                return color(0,0,0); //Ray absorbed
            }
        }
        else{
            vec3 unit_direction = unit_vector(current_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return current_attenuation * c;
        }
    }
    return color(0,0,0);
}

__device__ inline vec3 sample_square(curandState *rand_state) {
    return vec3(random_float(rand_state)-0.5f,random_float(rand_state)-0.5f, 0);
}

__device__ inline vec3 defocus_disk_sample(curandState *rand_state,camera* cam) {
    //Returns a random point in the camera defocus disk
    point3 p = random_in_unit_disk(rand_state);
    return cam->center + (p[0] * cam->defocus_disk_u) + (p[1] * cam->defocus_disk_v);
}

__device__ ray get_ray(curandState *rand_state,camera* cam,int i, int j) {
    //Construct a camera ray originating from the defocus disk
    //directed at a randomly sampled point around the pixel location i,j
    vec3 offset = sample_square(rand_state);
    point3 pixel_sample = cam->pixel00_loc 
                            + ((i+ offset.x()) * cam->pixel_delta_u) 
                            + ((j + offset.y()) *cam->pixel_delta_v);
    point3 ray_origin = (cam->defocus_angle <= 0) ? cam->center : defocus_disk_sample(rand_state,cam);
    vec3 ray_direction = pixel_sample - ray_origin;
    float ray_time =  random_float(rand_state); //Random time for motionblur

    return ray(ray_origin, ray_direction,ray_time);
}

__global__ void render_init(curandState *rand_state, camera *cam) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= cam->image_width) || (j >= cam->image_height)) return;
    int pixel_index = j*cam->image_width + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(curandState *rand_state,color *fb, hittable **world,camera *cam){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= cam->image_width) || (j >= cam->image_height)) return; //Due to block size, prevent rendering outside of image
    int pixel_index = j*cam->image_width + i;
    curandState local_rand_state = rand_state[pixel_index];
    color pixel_color(0,0,0);
    for(int s=0; s< cam->samples_per_pixel; s++){
        ray r = get_ray(&local_rand_state,cam,i,j);
        pixel_color += ray_color(&local_rand_state,cam,r, world);
    }
    fb[pixel_index] = pixel_color*cam->pixel_sample_scale;
}
#endif