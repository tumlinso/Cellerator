#pragma once
#include <array>
#include <stdexcept>
namespace ce_moon {
using Relation16=std::array<std::array<unsigned char,16>,16>;
inline Relation16 compose_relation(const Relation16& a,const Relation16& b){
    Relation16 c{};for(unsigned i=0;i<16;++i)for(unsigned j=0;j<16;++j){unsigned count=0;
        for(unsigned k=0;k<16;++k){if(a[i][k]>1||b[k][j]>1)throw std::invalid_argument("nonbinary relation");count+=a[i][k]*b[k][j];}c[i][j]=count!=0;}
    return c;
}
} // namespace ce_moon
