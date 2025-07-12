#ifndef AABB_CUH
#define AABB_CUH


class aabb{
    public:
        interval x,y,z;

        __device__ aabb() {} //Default empty interval
        __device__ aabb(const interval& x, const interval&y, const interval&z)
                        : x(x), y(y),z(z){}
        __device__ aabb(const point3& a, const point3& b){ //Point a and b as bounding box extrema
            x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]);
            y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
            z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);
        } 

        __device__ aabb(const aabb& box0, const aabb& box1){
            x = interval(box0.x, box1.x);
            y = interval(box0.y, box1.y);
            z = interval(box0.z, box1.z);
        }

        __device__ const interval& axis_interval(int n) const{
            if(n==1) return y;
            if(n==2) return z;
            return x;
        }

        __device__ bool hit(const ray&r, interval ray_t) const{
            const point3& ray_orig = r.origin();
            const vec3& ray_dir = r.direction();

            for (int axis=0; axis<3;axis++){
                const interval& ax = axis_interval(axis);
                const float adinv = 1.0f / ray_dir[axis];

                float t0 = (ax.min - ray_orig[axis]) * adinv;
                float t1 = (ax.max - ray_orig[axis]) * adinv;

                if (t0<t1){ 
                    if (t0>ray_t.min) ray_t.min = t0; 
                    if (t1<ray_t.max) ray_t.max = t1;
                } else{
                    if (t1>ray_t.min) ray_t.min = t1;
                    if (t0<ray_t.max) ray_t.max = t0;
                }

                if (ray_t.max <= ray_t.min) return false; //No intersection
            }
            return true; //Intersection
        }
};


#endif