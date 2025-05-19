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

class lambertian : public material{
    public:
    lambertian(const color& albedo): albedo(albedo){}

    bool scatter(
    const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
        const override{
            auto scatter_direction = rec.normal + random_unit_vector();

            //catch random vector = -normal degeneration
            if(scatter_direction.near_zero()){
                scatter_direction= rec.normal;
            }
            scattered = ray(rec.p,scatter_direction);
            attenuation = albedo;
            return true;

        }

    private:
    color albedo;
};
#endif