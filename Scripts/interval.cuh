#ifndef INTERVAL_CUH
#define INTERVAL_CUH

class interval {
    public:
    float min,max;
   __device__ interval() : min(+infinity),max(-infinity) {}
    __device__ interval(float min, float max) : min(min),max(max) {}

    __device__ float size() const {
        return max-min;
    }

    __device__ bool contains(float x) const {
        return min<= x && x <= max;
    }

    __device__ bool surrounds(float x) const {
        return min < x && x < max;
    }

    //Changed as cuda did not enjot memory qualifiers on data members
    __device__ static const interval empty(){
        return interval(+infinity, -infinity);
    }
    __device__ static const interval universe(){
        return interval(-infinity, +infinity);
    }

};

#endif