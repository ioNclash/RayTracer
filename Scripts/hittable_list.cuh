#ifndef HITTABBLE_LIST_CUH
#define HITTABLE_LIST_CUH

#include "aabb.cuh"
#include "hittable.cuh"


class hittable_list: public hittable{
    public:
    __device__ hittable_list(hittable** buffer,int capacity):size(0),capacity(capacity),list(buffer){}
       

    __device__ bool hit(const ray& r, interval ray_t, hit_record &rec) const override{
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;
        for(int i=0; i<size; i++){
            if (list[i] -> hit(r,interval(ray_t.min, closest_so_far),temp_rec)){
                hit_anything=true;
                closest_so_far= temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    __device__ void add(hittable* h){
        if(size<capacity){
            list[size++] = h;
            bbox = aabb(bbox,h->bounding_box());
        }
    }

    __device__ int get_size() const { return size; }
    __device__ int get_capacity() const { return capacity; }
    __device__ hittable* get_item(int i) const {
        if (i < 0 || i >= size) return nullptr; // Check bounds
        return list[i];
    }
    __device__ hittable** get_list() const { return list; }
    __device__ aabb bounding_box() const override { return bbox; }

    private:
        hittable **list;
        int size;
        int capacity;
        aabb bbox;


};

#endif