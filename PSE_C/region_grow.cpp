#include <iostream>
#include <queue>
#include <array>
#include <math.h>

#include "region_grow.h"

using namespace std;

namespace MY_LIB
{

struct Point
{
    int x;
    int y;
    double value[50];
};

double cal_dis(double *vec1, double *vec2, int depth)
{
    double sum = 0;
    for (int i = 0; i < depth; i++)
    {
        sum = (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]) + sum;
    }
    return sqrt(sum);
}


double cal_sim(double *vec1, double *vec2,int depth)
{
    double sum = 0;
    for (int i = 0; i < depth; i++)
    {
        sum = (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]) + sum;
    }

    return 2./(1+exp(sum));
}

void aver_vec(double *vec1, double w1, double *vec2, double w2, int depth, double *aver)
{
    for (int i = 0; i < depth; i++)
        aver[i] = w1 * vec1[i] + w2 * vec2[i];
}

Region::Region()
{
}

Region::~Region()
{
}

// shape [h,w,d]
// seed [y,x] (w,h)
// use DFS implement, or delete some code, we can use LBF 
void Region::region_grow(double *embed_vc,int *mask, int *shape, int *seed, double sim, int *flags,double *ins_vec,int &num)
{
    int DIR[8][2] = { {0, -1},{-1, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}};

    std::vector<Point> grow_stack;

    Point pt;
    pt.x=seed[1];
    pt.y=seed[0];
    copy(&embed_vc[(pt.y * shape[1] + seed[0]) * shape[2]], &embed_vc[(pt.y * shape[1] + seed[0]) * shape[2] + shape[2]], begin(pt.value));
    grow_stack.push_back(pt);
    flags[pt.y * shape[1] + pt.x] = 1;

    num = 0;
    double *move_aver = new double[shape[2]];
    std::fill_n(move_aver, shape[2], 0);

    while (!grow_stack.empty())
    {
        pt = grow_stack.back();
        // use WFS argrithom
        grow_stack.pop_back();

        bool end_flags=true;
        for (int i = 0; i < 8; i++)
        {
            Point nextpt;
            nextpt.x = pt.x + DIR[i][0];
            nextpt.y = pt.y + DIR[i][1];
            if (nextpt.x > -1 && nextpt.y > -1 && nextpt.x < shape[1] && nextpt.y < shape[0])
            {
                copy(&embed_vc[(nextpt.y * shape[1] + nextpt.x) * shape[2]], &embed_vc[(nextpt.y * shape[1] + nextpt.x) * shape[2] + shape[2]], begin(nextpt.value));
            }
            else
                continue;

            int has_visted = flags[nextpt.y * shape[1] + nextpt.x];
            int mask_value=mask[nextpt.y * shape[1] + nextpt.x];
            if (has_visted == 0&&mask_value>0)
            {
                end_flags=false;
                double *now_value = &embed_vc[(pt.y * shape[1] + pt.x) * shape[2]];
                double *next_value = &embed_vc[(nextpt.y * shape[1] + nextpt.x) * shape[2]];

                aver_vec(now_value, 0.9, move_aver, 0.1, shape[2], move_aver);

                // float similar = cal_sim(move_aver, next_value, shape[2]);
                float distance = cal_dis(move_aver, next_value, shape[2]);
                flags[nextpt.y * shape[1] + nextpt.x] = 1;
                if (distance < sim)
                // if (similar > sim)
                {
                    grow_stack.push_back(nextpt);
                    num++;
                    // break;
                }
            }
        }
        // // use DFS argrithom
        // if(end_flags==true)
        //     grow_stack.pop_back();
    }
    for (int i = 0; i < shape[2]; i++)
        ins_vec[i] = move_aver[i];
    delete[] move_aver;
}

} // namespace MY_LIB
