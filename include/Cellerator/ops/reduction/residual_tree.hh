#pragma once
#include <Cellerator/ops/sequence/lifting.hh>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace ce_moon {
struct ResidualTree {
    struct Node{std::size_t begin,end,left,right;double coarse,detail,maximum;bool leaf;};
    std::vector<Node> nodes;
    std::size_t root=0;
    explicit ResidualTree(const std::vector<double>& x){
        if(x.empty())throw std::invalid_argument("empty residual tree");
        for(double v:x)if(!std::isfinite(v))throw std::invalid_argument("nonfinite leaf");
        root=build(x,0,x.size());
    }
    std::size_t build(const std::vector<double>& x,std::size_t a,std::size_t z){
        if(z-a==1){nodes.push_back({a,z,0,0,x[a],0,x[a],true});return nodes.size()-1;}
        auto m=a+(z-a)/2,l=build(x,a,m),r=build(x,m,z);
        auto p=lift(nodes[l].coarse,nodes[r].coarse);
        nodes.push_back({a,z,l,r,p.coarse,p.detail,std::max(nodes[l].maximum,nodes[r].maximum),false});
        return nodes.size()-1;
    }
    double reconstruct(std::size_t leaf) const {
        if(leaf>=nodes[root].end)throw std::out_of_range("leaf");
        auto n=root;double coarse=nodes[n].coarse;
        while(!nodes[n].leaf){const auto& v=nodes[n];auto p=unlift({coarse,v.detail});
            if(leaf<nodes[v.left].end){coarse=p.first;n=v.left;}else{coarse=p.second;n=v.right;}}
        return coarse;
    }
    std::vector<std::size_t> above(double threshold,std::size_t* visited=nullptr)const{
        std::vector<std::size_t> out,stack{root};std::size_t count=0;
        while(!stack.empty()){auto i=stack.back();stack.pop_back();++count;const auto& v=nodes[i];
            if(v.maximum<=threshold)continue; // Certified ONLY for this scalar max query.
            if(v.leaf)out.push_back(v.begin);else{stack.push_back(v.right);stack.push_back(v.left);}}
        if(visited)*visited=count;return out;
    }
};
} // namespace ce_moon
