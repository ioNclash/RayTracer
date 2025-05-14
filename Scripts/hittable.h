#ifndef HITTABLE_H
#define HITTABLE_H



class hit_record{
    public:
    point3 p;
    vec3 normal;
    double t;
    bool front_face;

    void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0; //If > 0, 
        normal = front_face ? outward_normal : -outward_normal; //If outward normal is front facing keep positive, else make negative
    }
};

class hittable{
    public:
    virtual ~hittable() = default;//On destruction act as default

    virtual bool hit(const ray& r,double ray_tmin, double ray_tmax, hit_record& rec) const = 0;

};

#endif