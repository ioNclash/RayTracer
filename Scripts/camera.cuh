#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "hittable.cuh"
#include "material.cuh"

class camera{
    public:

    __host__ camera(
         float aspect_ratio = 16.0f/9.0f,
         int image_width = 1600,
         int samples_per_pixel=100,
         int max_depth=50,
         float fov=20.f,

         point3 look_from = point3(0,0,0),
         point3 look_at = point3(0,0,-1),
         vec3 vup = vec3(0,1,0),

        float defocus_angle = 0.0f,
        float focus_distance = 10.0f
        ) 
    {
        this->aspect_ratio = aspect_ratio;
        this->image_width = image_width;
        this->samples_per_pixel = samples_per_pixel;
        this->max_depth = max_depth;
        this->fov = fov;
        this->look_from = look_from;
        this->look_at = look_at;
        this->vup = vup;
        this->defocus_angle = defocus_angle;
        this->focus_distance = focus_distance;
        //Initialize camera parameters
        initialize();
    }

    //Getter
    __host__ __device__ inline const float& get_aspect_ratio() const {
        return this->aspect_ratio;
    }

    __host__ __device__ inline const int& get_image_width() const {
        return this->image_width;
    }
    __host__ __device__ inline const int& get_image_height() const {
        return this->image_height;
    }
    __host__ __device__ inline const int& get_samples_per_pixel() const {
        return this->samples_per_pixel;
    }
    __host__ __device__ inline const point3& get_center() const {
        return this->center;
    }
    __host__ __device__ inline const point3& get_pixel00_loc() const {
        return this->pixel00_loc;
    }
    __host__ __device__ inline  const vec3& get_pixel_delta_u() const {
        return this->pixel_delta_u;
    }
    __host__ __device__ inline const vec3& get_pixel_delta_v() const {
        return this->pixel_delta_v;
    }
    __host__ __device__ inline const float& get_pixel_sample_scale() const {
        return this->pixel_sample_scale;
    }


    __device__ color ray_color(curandState *rand_state,const ray&r, hittable **world) {
        ray current_ray = r;
        color current_attenuation = vec3(1.0f,1.0f,1.0f);
        for(int i=0; i<max_depth; i++){
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


    __device__ ray get_ray(int i, int j, curandState *rand_state) {
        //Construct a camera ray originating from the defocus disk
        //directed at a randomly sampled point around the pixel location i,j
        vec3 offset = sample_square(rand_state);
        point3 pixel_sample = pixel00_loc 
                              + ((i+ offset.x()) * pixel_delta_u) 
                              + ((j + offset.y()) *pixel_delta_v);
        point3 ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(rand_state);
        vec3 ray_direction = pixel_sample - ray_origin;
        return ray(ray_origin, ray_direction);
    }

    private:

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
    
    float fov ; //Field of view in degrees,
    point3 look_from ; //Camera position
    point3 look_at; //Point the camera is looking at
    vec3 vup; //Up direction for the camera
    vec3 u, v, w; //Camera coordinate system vectors

    float defocus_angle; //Variation angle of rays through each pixel
    float focus_distance; //Distance from camera look_from point to focus plane

    vec3 defocus_disk_u; //Defocus disk horizontal radius
    vec3 defocus_disk_v; //Defocus disk vertical radius


    __host__ void initialize(){
        //Image
        image_height = int(image_width/aspect_ratio);
        image_height = (image_height <1 ) ? 1 : image_height;   // image height

        pixel_sample_scale = 1.0f / float(samples_per_pixel);// Scale factor for pixel color accumulation

        center = look_from; // Set the camera center to the lookfrom position

        //Determine viewport dimensions
        float theta = degrees_to_radians(fov); // Convert field of view from degrees to radians
        float h = std::tan(theta/2);
        float viewport_height = 2.0f * h * focus_distance; // Height of the viewport
        float viewport_width = viewport_height * (float(image_width)/float(image_height)); //Viewport width based on aspect ratio
        
        //Calculate camera coordinate system
        w = unit_vector(look_from - look_at); // Direction from camera to lookat point
        u = unit_vector(cross(vup, w)); // Perpendicular vector in the x direction
        v = cross(w, u); // Perpendicular vector in the y direction

        //Calculate vectors across viewport edges
        vec3 viewport_u = viewport_width * u; // Width vector of the viewport
        vec3 viewport_v = viewport_height * v; // Height vector of the viewport

        //Calculate pixel delta
        pixel_delta_u = viewport_u/image_width;
        pixel_delta_v = viewport_v/image_height;

        //Calculate location of bottom left pixel
        point3 viewport_bottom_left = center - (focus_distance*w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_bottom_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        float defocus_radius = focus_distance * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;

    }
    __device__ vec3 sample_square(curandState *rand_state) const{
        return vec3(random_float(rand_state)-0.5f,random_float(rand_state)-0.5f, 0);
    }

    __device__ vec3 defocus_disk_sample(curandState *rand_state) const{
        //Returns a random point in the camera defocus disk
        point3 p = random_in_unit_disk(rand_state);
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

};



__global__ void render_init(camera *cam, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= cam->get_image_width()) || (j >= cam->get_image_height())) return;
    int pixel_index = j*cam->get_image_width() + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(color *fb, hittable **world,camera *cam, curandState *rand_state){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= cam->get_image_width()) || (j >= cam->get_image_height())) return; //Due to block size, prevent rendering outside of image
    int pixel_index = j*cam->get_image_width() + i;
    curandState local_rand_state = rand_state[pixel_index];
    color pixel_color(0,0,0);
    for(int s=0; s< cam->get_samples_per_pixel(); s++){
        ray r = cam->get_ray(i,j,&local_rand_state);
        pixel_color += cam->ray_color(&local_rand_state,r, world);
    }
    fb[pixel_index] = pixel_color*cam->get_pixel_sample_scale();
}
#endif