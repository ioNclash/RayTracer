#ifndef HITTABLE_CUH
#define HITTABLE_CUH

struct hit_record{
    float t;
    vec3 p;
    vec3 normal;
    bool front_face;

    __device__ void set_face_normal(const ray&r, const vec3& outward_normal){
        front_face = dot(r.direction(), outward_normal) <0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
    public:
        __device__ virtual bool 
        hit(const ray& r, interval ray_t,hit_record& rec) const = 0;
};

#endif