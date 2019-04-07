#include <iostream>
#include <queue>
#include <array>
#include <set>

#include "Expand.h"

using namespace std;
using namespace MY_LIB;

Expand::Expand() 
{
}

Expand::~Expand() 
{
}

void Expand::expansion(int* CC, int* Si,int h,int w) 
{
    std::queue<std::array<int,3>> Q;
    std::set<std::array<int,3>> T;     // point and label
    std::set<std::array<int,2>> P;     //Point

    for(int y=0;y<h;y++)
    {
        for(int x=0;x<w;x++)
        {
            std::array<int,3> t_temp={y,x,CC[w*y+x]};
            if(CC[w*y+x]>0)
            {   
                std::array<int,2> p_temp={y,x};
                T.insert(t_temp);
                P.insert(p_temp);
                Q.push(t_temp);
            }
            
        }
    }   
    while (!Q.empty())
    {
        std::array<int,3> t_temp=Q.front();
        Q.pop();
        for(int y=t_temp.at(0)-1;y<t_temp.at(0)+2;y++)
        {
            for(int x=t_temp.at(1)-1;x<t_temp.at(1)+2;x++)
            {
                if(y>=0&&y<h&&x>=0&&x<w)
                {
                    std::array<int,2> p_temp={y,x};
                    if(P.find(p_temp)== P.end()&&Si[y*w+x]==1)
                    {
                        P.insert(p_temp);
                        std::array<int,3> t_new={y,x,t_temp.at(2)};
                        T.insert(t_new);
                        Q.push(t_new);
                        CC[y*w+x]=t_temp.at(2);
                        
                    }
                }
            }
        }
    }
    //  for(int y=0;y<h;y++)
    // {
    //     for(int x=0;x<w;x++)
    //     {
    //         cout<<CC[y*w+x]<<endl;
    //     }
    // }
    return ;
}