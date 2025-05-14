#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <memory>
#include <vector>

using std::make_shared;
using std::shared_ptr;

class hittable_list: public hittable{
    public:
    std::vector<shared_ptr<hittable>> objects; //Dynamic Resizing Pointer that stores hittable objects, named object

    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) {add(object);} 

    void clear() {objects.clear();} //Clear hittable object list

    void add(shared_ptr<hittable> object){
        objects.push_back(object);
    }

    bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const override{
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_tmax;

        for (const auto& object: objects){
            if (object->hit(r, ray_tmin,closest_so_far,temp_rec)){ //Check for all objects in list if there is a hit, and save closest hit point
                hit_anything=true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

    }

};

#endif