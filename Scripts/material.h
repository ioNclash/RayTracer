#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"

class material{ //Produce scattered ray or state absorbed, if scattered state ray attenuation
    public:
    virtual ~material() = default;

    virtual bool scatter(
    const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
        const{
            return false;
        }
    
};


#endif