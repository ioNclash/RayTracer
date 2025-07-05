#ifndef HITTABBLE_LIST_CUH
#define HITTABLE_LIST_CUH

#include "aabb.cuh"


class hittable_list: public hittable{
    public:
    __device__ hittable_list() {}
    __device__ hittable_list(hittable **l, int n) {list =l; list_size=n;}
    __device__ bool hit(const ray& r, interval ray_t, hit_record &rec) const override{
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;
        for(int i=0; i<list_size; i++){
            if (list[i] -> hit(r,interval(ray_t.min, closest_so_far),temp_rec)){
                hit_anything=true;
                closest_so_far= temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }
    hittable **list;
    int list_size;


};

#endif