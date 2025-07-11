#ifndef VEC3_CUH
#define VEC3_CUH

class vec3{
    public:
    //Vec3 Float Array
    float e[3];

    //Constructors
    __host__ __device__ inline vec3() {}
    __host__ __device__ inline vec3 (float e0, float e1, float e2) : e{e0,e1,e2} {}

    //Get Components
    __host__ __device__ inline float x() const {return e[0];}
    __host__ __device__ inline float y() const {return e[1];}
    __host__ __device__ inline float z() const {return e[2];}

    //Operators
    __host__ __device__ inline vec3 operator-() const {return vec3{-e[0],-e[1],-e[2]};}
    __host__ __device__ inline float operator[](int i) const {return e[i];}
    __host__ __device__ inline float& operator[](int i) {return e[i];}

    __host__ __device__ inline vec3& operator+=(const vec3& v){
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

     __host__ __device__ inline vec3& operator*=(float t){
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ inline vec3& operator*=(vec3& v){
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }


    __host__ __device__ inline vec3& operator/=(float t) {
        return *this *= 1/t;
    }

    //Length
    __host__ __device__ inline float length() const{
        return std::sqrt(length_squared());
    }

    __host__ __device__ inline float length_squared() const{
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2] ;
    }

    __device__ bool near_zero() const{
        // Return true if the vector is close to zero in all dimensions
        float s = 1e-8f; // A small value to avoid division by zero
        return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
    }

    //Device only cuda random vectors
    __device__ static vec3 random(curandState *random_state) {
        return vec3(random_float(random_state), 
                    random_float(random_state), 
                    random_float(random_state));

    }

    __device__ static vec3 random(curandState *random_state, float min, float max) {
        return vec3(min + (max - min) * random_float(random_state),
                    min + (max - min) * random_float(random_state),
                    min + (max - min) * random_float(random_state));
    }


    
};

using point3 = vec3; //Alias

// Vector Utility Functions

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, float t) {
    return (1/t) * v;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

__device__ inline vec3 random_in_unit_disk(curandState *random_state){
    while(true){
        point3 p = point3(random_float(random_state,-1.0f,1.0f), 
                          random_float(random_state,-1.0f,1.0f),
                          0.0f);
        if(p.length_squared() < 1.0f) {
            return p;
        }
    }
}

__device__ inline vec3 random_unit_vector(curandState *random_state){
    while(true){
        point3 p = point3::random(random_state, -1.0f, 1.0f);
        float lensq = p.length_squared();
        if(1e-30f<=lensq<= 1){ //1e-30f is a small value to avoid division by zero
            return p/ sqrt(lensq);
        }
    }

}

__device__ inline vec3 random_on_hemisphere(curandState *random_state,const vec3& normal){
    vec3 on_unit_sphere = random_unit_vector(random_state);
    if(dot(on_unit_sphere, normal) > 0.0f) {
        return on_unit_sphere; // In the same hemisphere
    } else {
        return -on_unit_sphere; // In the opposite hemisphere
    }
}

__device__ inline vec3 reflect(const vec3& v, const vec3& normal) {
    return v - 2 * dot(v, normal) * normal;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& normal, float etai_over_etat){
    float cos_theta = std::fmin(dot(-uv,normal), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv +cos_theta*normal);
    vec3 r_out_parallel = -std::sqrt(std::fabs(1.0f - r_out_perp.length_squared())) * normal;
    return r_out_perp + r_out_parallel;
}

#endif
