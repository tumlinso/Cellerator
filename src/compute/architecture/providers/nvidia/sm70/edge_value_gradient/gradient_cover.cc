#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/gradient_cover.hh>
#include <algorithm>
#include <limits>
#include <new>
namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient {
namespace {
struct entry { std::uint64_t key=0; std::uint32_t value=absent_slot; };
std::uint64_t key(std::uint32_t s,std::uint32_t d) { return (std::uint64_t(s)<<32)|d; }
std::size_t locate(std::vector<entry>& table,std::uint64_t k) {
    std::uint64_t h=k;h^=h>>30;h*=0xbf58476d1ce4e5b9ULL;h^=h>>27;h*=0x94d049bb133111ebULL;h^=h>>31;
    std::size_t i=h&(table.size()-1);
    while(table[i].value!=absent_slot && table[i].key!=k)i=(i+1)&(table.size()-1);
    return i;
}
}
cover_status prepare_gradient_cover(const gradient_edge *edges,std::uint64_t count,
    std::uint32_t sources,std::uint32_t destinations,const tile_hint *hints,
    std::uint64_t hint_count,std::uint64_t limit,gradient_cover &output) noexcept {
    output={};
    if(count>=absent_slot || (count&&!edges) || (hint_count&&!hints)
        || hint_count>absent_slot/256u)return cover_status::invalid_argument;
    std::uint64_t table_count=1;while(table_count<count*2)table_count*=2;
    const std::uint64_t max_tiles=hint_count?hint_count:count/128u;
    if (max_tiles > absent_slot / 256u) return cover_status::capacity_exceeded;
    // Two fixed open-address tables, edge flags/maps, worst-case residual and
    // tile storage; no node allocator or Cartesian group census.
    const std::uint64_t bound=2*table_count*sizeof(entry)+count*(sizeof(gradient_edge)+5u)
        +max_tiles*sizeof(gradient_tile);
    if(limit && bound>limit)return cover_status::capacity_exceeded;
    try {
        std::vector<entry> lookup(table_count),groups(table_count);
        std::vector<unsigned char> seen(count,0);
        gradient_cover result;result.tiles.reserve(max_tiles);result.residual.reserve(count);
        result.edge_to_tile_slot.assign(count,absent_slot);result.preparation_byte_bound=bound;
        for(std::uint32_t i=0;i<count;++i){const auto e=edges[i];
            if(e.source>=sources||e.destination>=destinations||e.physical>=count||seen[e.physical])return cover_status::invalid_argument;
            seen[e.physical]=1;auto &v=lookup[locate(lookup,key(e.source,e.destination))];
            if(v.value!=absent_slot)return cover_status::invalid_argument;
            v={key(e.source,e.destination),i};
            auto &g=groups[locate(groups,key(e.source/16,e.destination/16))];
            if(g.value==absent_slot)g={key(e.source/16,e.destination/16),0};
            ++g.value;
        }
        auto add=[&](const tile_hint &hint){
            if(!hint.source_count||hint.source_count>16||!hint.destination_count||hint.destination_count>16)return false;
            for(unsigned i=0;i<hint.source_count;++i){if(hint.sources[i]>=sources)return false;for(unsigned j=0;j<i;++j)if(hint.sources[i]==hint.sources[j])return false;}
            for(unsigned i=0;i<hint.destination_count;++i){if(hint.destinations[i]>=destinations)return false;for(unsigned j=0;j<i;++j)if(hint.destinations[i]==hint.destinations[j])return false;}
            gradient_tile tile;tile.gather=hint;tile.physical_slots.fill(absent_slot);
            unsigned occupied=0;
            for(unsigned s=0;s<hint.source_count;++s)for(unsigned d=0;d<hint.destination_count;++d){
                auto v=lookup[locate(lookup,key(hint.sources[s],hint.destinations[d]))];if(v.value==absent_slot)continue;
                auto p=edges[v.value].physical;if(result.edge_to_tile_slot[p]!=absent_slot)return false;
                auto slot=s*16+d;result.edge_to_tile_slot[p]=std::uint32_t(result.tiles.size()*256+slot);tile.physical_slots[slot]=p;++occupied;
            }
            if(occupied)result.tiles.push_back(tile);
            return true;
        };
        if(hint_count){for(std::uint64_t i=0;i<hint_count;++i)if(!add(hints[i]))return cover_status::invalid_argument;}
        else for(auto g:groups)if(g.value!=absent_slot&&g.value>=128){
            tile_hint h;auto s=std::uint32_t(g.key>>32)*16u,d=std::uint32_t(g.key)*16u;
            h.source_count=std::min(16u,sources-s);h.destination_count=std::min(16u,destinations-d);
            for(unsigned i=0;i<h.source_count;++i)h.sources[i]=s+i;
            for(unsigned i=0;i<h.destination_count;++i)h.destinations[i]=d+i;
            if(!add(h))return cover_status::invalid_argument;
        }
        for(std::uint32_t i=0;i<count;++i)if(result.edge_to_tile_slot[edges[i].physical]==absent_slot)result.residual.push_back(edges[i]);
        result.persistent_bytes=result.tiles.capacity()*sizeof(gradient_tile)
            +result.residual.capacity()*sizeof(gradient_edge)+result.edge_to_tile_slot.capacity()*sizeof(std::uint32_t);
        output=std::move(result);return cover_status::success;
    }catch(const std::bad_alloc&){return cover_status::capacity_exceeded;}
}
}
