// Independent host mathematics witness, not an accelerator provider.
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
    struct edge { unsigned source, destination; double weight; };
    // Non-square, signed, empty destination 1, repeated endpoints at distinct edges.
    const std::array<edge, 5> edges{{{0,0,2}, {2,0,-1}, {1,2,3}, {0,3,-2}, {0,0,0.5}}};
    const std::array<double,3> x{{1,2,4}};
    const std::array<double,4> u{{2,7,-1,3}};
    std::array<double,4> y{};
    std::array<double,3> z{};
    for (const auto& e : edges) y[e.destination] += e.weight*x[e.source];
    // Independent grouping by source rather than reuse of forward index traversal.
    for (unsigned source=0; source<z.size(); ++source)
        for (const auto& e : edges)
            if (e.source==source) z[source] += e.weight*u[e.destination];
    const std::array<double,4> expected_y{{-1.5,0,6,-2}};
    const std::array<double,3> expected_z{{-1,-3,-2}};
    if (y!=expected_y || z!=expected_z) return 1;
    double lhs=0, rhs=0;
    for(unsigned i=0;i<y.size();++i) lhs+=y[i]*u[i];
    for(unsigned i=0;i<z.size();++i) rhs+=x[i]*z[i];
    if(lhs!=rhs || lhs!=-15) return 2;
    // Source permutation preserves biological result only with matching remap.
    const std::array<unsigned,3> new_slot{{2,0,1}};
    std::array<double,3> permuted{};
    for(unsigned i=0;i<x.size();++i) permuted[new_slot[i]]=x[i];
    std::array<double,4> recovered{}, wrong{};
    for(const auto& e:edges) {
        recovered[e.destination]+=e.weight*permuted[new_slot[e.source]];
        wrong[e.destination]+=e.weight*permuted[e.source];
    }
    if(recovered!=expected_y || wrong==expected_y) return 3;
    // Dropping the distinct duplicate edge changes the mathematics.
    auto dropped=y; dropped[0]-=edges.back().weight*x[0];
    if(dropped==expected_y) return 4;
    std::cout << "host exact-support duality and order controls passed\n";
}
