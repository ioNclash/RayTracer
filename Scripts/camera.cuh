#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "hittable.cuh"
#include "material.cuh"

class camera{
    public:

    __host__ camera(float aspect_ratio = 16.0f/9.0f, int image_width = 1600) {
        this->aspect_ratio = aspect_ratio;
        this->image_width = image_width;
        initialize();
    }

    //Setters
    __host__ inline void set_aspect_ratio(float aspect_ratio){
        this->aspect_ratio = aspect_ratio;
        initialize();
    }

    __host__ inline void set_image_width(int image_width){
        this->image_width = image_width;
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
    __host__ __device__ inline const int& get_sample_count() const {
        return this->sample_count;
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


    __device__ color ray_color(curandState *rand_state,const ray&r, hittable **world) {
        ray current_ray = r;
        color current_attenuation = vec3(1.0f,1.0f,1.0f);
        for(int i=0; i<sample_count; i++){
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
        //Generate a ray from the camera to the pixel at (i,j)
        vec3 offset = sample_square(rand_state);
        point3 pixel_sample = pixel00_loc 
                              + ((i+ offset.x()) * pixel_delta_u) 
                              + ((j + offset.y()) *pixel_delta_v);
        point3 ray_origin = center;
        vec3 ray_direction = pixel_sample - ray_origin;
        return ray(ray_origin, ray_direction);
    }

    private:

    float aspect_ratio;
    int image_width;
    int image_height;
    int sample_count;
    point3 center;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    int max_depth = 50;
    __host__ void initialize(){
        //Image
        image_height = int(image_width/aspect_ratio);
        image_height = (image_height <1 ) ? 1 : image_height;   // image height

        //Camera
        float focal_length = 1.0f;
        float viewport_height = 2.0f;
        float viewport_width = viewport_height * (float(image_width)/float(image_height));
        center = point3(0,0,0);
        sample_count = 50;

        //Calculate vectors across viewport edges
        vec3 viewport_u = vec3(viewport_width,0,0);
        vec3 viewport_v = vec3(0,viewport_height,0); //+ here to make pixel00 the bottom left pixel

        pixel_delta_u = viewport_u/image_width;
         pixel_delta_v = viewport_v/image_height;

        point3 viewport_bottom_left = center - vec3(0,0,focal_length) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_bottom_left + 0.5f * (pixel_delta_u + pixel_delta_v);

    }
    __device__ vec3 sample_square(curandState *rand_state) const{
        return vec3(random_float(rand_state)-0.5f,random_float(rand_state)-0.5f, 0);
    }

};



__global__ void render_init(camera *cam, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= cam->get_image_width()) | (j >= cam->get_image_height())) return;
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
    for(int s=0; s< cam->get_sample_count(); s++){
        ray r = cam->get_ray(i,j,&local_rand_state);
        pixel_color += cam->ray_color(&local_rand_state,r, world);
    }
    fb[pixel_index] = pixel_color*1/(cam->get_sample_count());
}
#endif