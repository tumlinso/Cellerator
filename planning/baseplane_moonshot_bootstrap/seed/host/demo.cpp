#include <bp_moon/reference.hpp>
#include <iostream>
using namespace bp_moon;
int main(){try{
    const std::string dna="ACGTACGTTTTTACGNACGTACGTGGGGACGT";
    std::vector<double> scalar;
    for(std::size_t i=0;i<dna.size();i+=4){
        auto s=dna.substr(i,4); // Fixed fixture windows, not a learned segmentation claim.
        auto effect=summarize_sequence(s);scalar.push_back(double(effect.to[0]));
    }
    ResidualTree hierarchy(scalar);std::size_t visited=0;auto requests=hierarchy.above(8.,&visited);
    ExactInterner dictionary;std::vector<Keyed> keys;
    for(std::size_t i=0;i<scalar.size();++i)keys.push_back({dictionary.intern(dna.substr(4*i,4)),i});
    const auto peers=rendezvous(keys);
    std::cout<<"Exact input: "<<dna<<"\n";
    std::cout<<"Toy regional effects: "<<scalar.size()<<"; distinct exact strings: "<<peers.size()<<"\n";
    std::cout<<"Query visits "<<visited<<" tree nodes and refines "<<requests.size()<<" windows.\n";
    for(auto id:requests)std::cout<<"["<<id*4<<","<<std::min(dna.size(),id*4+4)<<") "<<dna.substr(id*4,4)<<" effect="<<hierarchy.reconstruct(id)<<"\n";
    std::cout<<"Demonstrates composition/provenance/refinement plumbing only; not learned biology or a speed result.\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
