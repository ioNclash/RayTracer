#ifndef SPHERE_CUH
#define SPHERE_CUH



class sphere: public hittable {
    public:
    material *mat_ptr;

    //Stationary Sphere Constructor
    __device__ sphere(const point3& static_center, float radius,material *m):
               center(static_center, vec3(0,0,0)), radius(std::fmaxf(0,radius)),mat_ptr(m)
               {
                 vec3 rvec = vec3(radius,radius,radius);
                 bbox = aabb(static_center -rvec,static_center + rvec);

               }
    
    //Moving Sphere Constructor
    __device__ sphere(const point3& start_center, const point3& end_center, float radius,material*m):
                center(start_center,end_center-start_center),radius(std::fmaxf(0,radius)),mat_ptr(m)
                {
                 vec3 rvec = vec3(radius,radius,radius);
                 aabb box1(center.at(0) - rvec, center.at(0) + rvec);
                 aabb box2(center.at(1) - rvec, center.at(1) + rvec);
                 bbox = aabb(box1, box2);
                }
    
    __device__ bool
    hit(const ray&r,interval ray_t, hit_record& rec)
    const override{
        point3 current_center = center.at(r.time());
        vec3 oc = current_center - r.origin();
        float a = r.direction().length_squared();
        float h = dot(r.direction(),oc);
        float c = oc.length_squared() - radius*radius;

        float discriminant = h*h -a*c;
        if(discriminant < 0)
            return false;
        
        float sqrtd = std::sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        float root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p-current_center) /radius;
        rec.set_face_normal(r,outward_normal);
        rec.mat_ptr = mat_ptr;

        return true;

    }

    __device__ aabb bounding_box() const override {
        return bbox;
    }
    
    private:
    ray center; //Center point and direction vector for moving spheres 
    float radius;
    aabb bbox; //Bounding box for the sphere

};

#endif