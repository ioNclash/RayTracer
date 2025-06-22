#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "hittable.cuh"

class material{
    public:
    __device__ virtual bool scatter(
        curandState *rand_state, const ray& r_in,
        const hit_record& rec, color& attenuation,ray& scattered)
    const {
        return false;
    }
};

class lambertian : public material {
    public:
    __device__ lambertian(const color& albedo) : albedo(albedo){}
    __device__ virtual bool scatter(
        curandState *rand_state, const ray& r_in,
        const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);

        // Catch degenerate scatter direction
        if(scatter_direction.near_zero()){
            scatter_direction = rec.normal;
        }

        scattered = ray(rec.p,scatter_direction,r_in.time());
        attenuation = albedo;
        return true;
    }
    private:
    color albedo;
};

class metal : public material {
    public:
    __device__ metal(const color& albedo,float fuzz) : albedo(albedo),fuzz(fuzz < 1 ? fuzz : 1) {}
    __device__ virtual bool scatter(
        curandState *rand_state, const ray& r_in,
        const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz * random_unit_vector(rand_state));
        scattered = ray(rec.p,reflected,r_in.time());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
    private:
    color albedo;
    float fuzz;
};

class dielectric : public material{
    public:
    __device__ dielectric(float refraction_index) : refraction_index(refraction_index) {}

    __device__ virtual bool scatter(curandState *rand_state,const ray& r_in, const hit_record& rec,
    color& attenuation, ray& scattered)
    const override{
        attenuation = color(1.0f,1.0f,1.0f);
        float ri = rec.front_face ? (1.0f / refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = std::fmin(dot(-unit_direction, rec.normal),1.0f);
        float sin_theta = std::sqrt(1.0f - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0f;
        vec3 direction;
        
        if (cannot_refract ||reflectance(cos_theta,ri) > random_float(rand_state)){
            direction = reflect(unit_direction,rec.normal);
        }
        else{
            direction = refract(unit_direction,rec.normal,ri);
        }

        scattered = ray(rec.p, direction,r_in.time());
        return true;

    }                    

    private:
    float refraction_index;
    __device__ static float reflectance(float cosine, float refraction_index){
        float r0 = (1-refraction_index) / (1+refraction_index);
        r0 = r0 * r0;
        return r0 + (1-r0)*std::pow((1-cosine),5.0f);
    }
}; 

#endif