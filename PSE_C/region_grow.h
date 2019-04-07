#ifndef REGION_GROW_H
#define REGION_GROW_H

namespace MY_LIB {
class Region
{
  public:
    Region();
    ~Region();
    void region_grow(double* embed_vc,int *mask,int* shape,int* seed, double delta,int* flags,double* ins_vec,int &num);
};
}

#endif