#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "hittable.cuh"

struct camera_data{
    float aspect_ratio;
    int image_width;
    int image_height;
    point3 center;
    point3 pixel00_loc;
    vec3 pixel_width;
    vec3 pixel_height;


};

class camera{
    public:

    __host__ camera(float aspect_ratio=16.0f/9.0f,float image_width=1600){
        //Image Defaults
        cam_data.aspect_ratio = aspect_ratio;
        cam_data.image_width = image_width;
        initialize();

    }

    //Setters
    __host__ void set_aspect_ratio(float aspect_ratio){
        cam_data.aspect_ratio = aspect_ratio;
        initialize();
    }

    __host__ void set_image_width(int image_width){
        cam_data.image_width = image_width;
        initialize();
    }

    //Getter
    __host__ const camera_data& get_camera_data() const {
        return cam_data;
    }



    private:
    camera_data cam_data;
    __host__ void initialize(){
        //Image
        cam_data.image_height = int(cam_data.image_width/cam_data.aspect_ratio);
        cam_data.image_height = (cam_data.image_height <1 ) ? 1 : cam_data.image_height;   // image height

        //Camera
        float focal_length = 1.0f;
        float viewport_height = 2.0f;
        float viewport_width = viewport_height * (float(cam_data.image_width)/float(cam_data.image_height));
        cam_data.center = point3(0,0,0);

        //Calculate vectors across viewport edges
        vec3 viewport_u = vec3(viewport_width,0,0);
        vec3 viewport_v = vec3(0,viewport_height,0); //+ here to make pixel00 the bottom left pixel

        cam_data.pixel_width = viewport_u/cam_data.image_width;
        cam_data. pixel_height = viewport_v/cam_data.image_height;

        point3 viewport_bottom_left = cam_data.center - vec3(0,0,focal_length) - viewport_u/2 - viewport_v/2;
        cam_data.pixel00_loc = viewport_bottom_left + 0.5f * (cam_data.pixel_width + cam_data.pixel_height);

    }


};

__device__ color ray_color(const ray&r, hittable **world) {
            hit_record rec;
            if ((*world)->hit(r,interval(0,infinity),rec)){
                return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
            }
            else {
            vec3 unit_direction = unit_vector(r.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
        }
    }

__global__ void render(color *fb, hittable **world,camera_data *cam_data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= cam_data->image_width) || (j >= cam_data->image_height)) return; //Due to block size, prevent rendering outside of image

    point3 pixel_center = cam_data->pixel00_loc + (i*cam_data->pixel_width) + (j*cam_data->pixel_height);
    vec3 ray_direction = pixel_center - cam_data->center;
    ray r(cam_data->center,ray_direction);
    int pixel_index = j*cam_data->image_width + i; //get index in frame buffer
        fb[pixel_index] = ray_color(r,world);
}
#endif