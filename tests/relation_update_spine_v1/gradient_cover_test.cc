#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/gradient_cover.hh>
#include <algorithm>
#include <cassert>
#include <cstdio>
using namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient;
void exact(const std::vector<gradient_edge>& e,const gradient_cover& c){
    std::vector<unsigned> n(e.size());
    for(unsigned t=0;t<c.tiles.size();++t)for(unsigned slot=0;slot<256;++slot){auto p=c.tiles[t].physical_slots[slot];if(p==absent_slot)continue;
        assert(p<n.size());++n[p];assert(c.edge_to_tile_slot[p]==t*256+slot);
        auto it=std::find_if(e.begin(),e.end(),[&](auto x){return x.physical==p;});assert(it!=e.end());
        assert(it->source==c.tiles[t].gather.sources[slot/16]);assert(it->destination==c.tiles[t].gather.destinations[slot%16]);}
    for(auto x:c.residual){++n[x.physical];assert(c.edge_to_tile_slot[x.physical]==absent_slot);}
    for(auto x:n)assert(x==1);
}
int main(){
    std::vector<gradient_edge> e;for(unsigned s=0;s<16;++s)for(unsigned d=0;d<16;++d)if(s!=3||d!=5)e.push_back({s,d,unsigned(e.size())});
    for(unsigned i=0;i<6;++i)e.push_back({20+i,31-i,unsigned(e.size())});
    std::reverse(e.begin(),e.end());gradient_cover c;
    assert(prepare_gradient_cover(e.data(),e.size(),32,32,nullptr,0,100000,c)==cover_status::success);
    assert(c.tiles.size()==1&&c.residual.size()==6);exact(e,c);
    assert(c.persistent_bytes<=c.preparation_byte_bound);
    assert(prepare_gradient_cover(e.data(),e.size(),32,32,nullptr,0,1,c)==cover_status::capacity_exceeded);
    tile_hint h;h.source_count=h.destination_count=16;for(unsigned i=0;i<16;++i){h.sources[i]=15-i;h.destinations[i]=(i*7)%16;}
    assert(prepare_gradient_cover(e.data(),e.size(),32,32,&h,1,100000,c)==cover_status::success);exact(e,c);
    tile_hint hints[]={h,h};assert(prepare_gradient_cover(e.data(),e.size(),32,32,hints,2,100000,c)==cover_status::invalid_argument);
    e={{3,2,1},{30,31,0}};assert(prepare_gradient_cover(e.data(),2,32,32,nullptr,0,100000,c)==cover_status::success);assert(c.tiles.empty()&&c.residual.size()==2);exact(e,c);
    e[1]={3,2,0};assert(prepare_gradient_cover(e.data(),2,32,32,nullptr,0,100000,c)==cover_status::invalid_argument);
    e[1]={4,2,1};assert(prepare_gradient_cover(e.data(),2,32,32,nullptr,0,100000,c)==cover_status::invalid_argument);
    assert(prepare_gradient_cover(nullptr,0,0,0,nullptr,0,100000,c)==cover_status::success);assert(c.tiles.empty()&&c.residual.empty());
    puts("gradient cover exact support/inverse maps/budget PASS");
}
