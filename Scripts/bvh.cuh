#ifndef BVH_CUH
#define BVH_CUH

#include "aabb.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"

class bvh_node : public hittable {
    public:
        __device__ bvh_node(hittable_list list): bvh_node(list.get_list(),0,list.get_size()) {

        }
        __device__ bvh_node(hittable** objects, int start, int end) {
                //TODO implent later
        }
    

    __device__ bool hit(const ray& r,interval ray_t, hit_record& rec) const override{
        if(!bbox.hit(r,ray_t)) return false; //No intersection with bounding box

        bool hit_left = left->hit(r, ray_t, rec);
        bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

        return hit_left || hit_right;
    }

    __device__ aabb bounding_box() const override { return bbox; }

    private:
    hittable* left;
    hittable* right;
    aabb bbox;
};

#endif