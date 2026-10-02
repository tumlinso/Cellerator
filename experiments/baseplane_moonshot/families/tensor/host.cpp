#include <ce_moon/tensor.hpp>
#include <iostream>
#include <stdexcept>
using namespace ce_moon::tensor;
void check(bool x){if(!x)throw std::runtime_error("tensor fixture mismatch");}
int main(){
 Matrix16 x{},w{};Ids16 ids{};for(unsigned i=0;i<16;++i){ids[i]=1000+i;x[i*16]=float(i+1);x[i*16+1]=2;w[i*16+i]=1;}
 auto identity=object_features(x,w,{16,16,16},ids);check(identity.values==x&&identity.ids==ids);
 w={};w[0]=2;w[16]=3;w[1]=-1;auto transformed=object_features(x,w,{3,2,2},ids);check(transformed.values[0]==8&&transformed.values[1]==-1&&transformed.values[32]==12&&transformed.values[48]==0&&transformed.ids[3]==0);
 // Independent two-feature directed score oracle, distinct source tokens.
 Matrix16 q{},k{};q[0]=1;q[1]=2;q[16]=3;q[17]=-1;k[0]=2;k[1]=0;k[16]=-1;k[17]=4;
 auto scores=relation_scores(q,k,2,2);check(scores[0]==2&&scores[1]==7&&scores[16]==6&&scores[17]==-7);
 ids[0]=7;ids[1]=9001;PairMask mask{};mask[1]=mask[16]=1;auto pairs=compact_relations(scores,ids,2,mask,0,1);check(pairs.required==2&&pairs.overflow&&pairs.pairs[0].from==7&&pairs.pairs[0].to==9001);check(compact_relations(scores,ids,2,mask,6.5f,4).required==1);
 ce_moon::Relation16 a{},b{},expected{};a[0][1]=a[0][2]=1;b[1][3]=b[2][3]=1;b[2][4]=1;expected[0][3]=expected[0][4]=1;
 auto counts=finite_relation_counts(a,b);check(counts[3]==2&&counts[4]==1&&counts[255]==0&&existence(counts)==expected&&existence(counts)==ce_moon::compose_relation(a,b));
 a={};b={};for(auto& row:a)row.fill(1);for(auto& row:b)row.fill(1);for(float c:finite_relation_counts(a,b))check(c==16);
 Matrix16 states{},weights{};states[0]=-1;states[16]=1;weights[0]=1;auto table=possible_states(states,weights,{2,1,1});std::array<float,16> coordinates{};coordinates[0]=-1;coordinates[1]=1;
 auto sampled=interpolate(table,coordinates,2,1,1);check(sampled.values[0]==table[16]);auto query=interpolate(table,coordinates,2,1,.5f);float error=std::abs(query.values[0]-std::tanh(.5f));check(error>.08f&&error<.09f);
 // Valid float endpoints must not overflow subtraction in interpolation.
 const float largest=std::numeric_limits<float>::max();
 Matrix16 extreme{};extreme[16]=1;
 std::array<float,16> extremes{};extremes[0]=-largest;extremes[1]=largest;
 auto midpoint=interpolate(extreme,extremes,2,1,0);check(midpoint.alpha==.5f&&midpoint.values[0]==.5f);
 extremes[0]=0;extremes[1]=1;extreme[0]=-largest;extreme[16]=largest;
 check(interpolate(extreme,extremes,2,1,.5f).values[0]==0);
 unsigned rejected=0;try{coordinates[1]=-1;interpolate(table,coordinates,2,1,0);}catch(const std::invalid_argument&){++rejected;}try{compact_relations(scores,ids,0,mask,0,1);}catch(const std::invalid_argument&){++rejected;}try{a[0][0]=2;finite_relation_counts(a,b);}catch(const std::invalid_argument&){++rejected;}check(rejected==3);
 std::cout<<"E21 identity/nontrivial/padding passed; E22 directed mask/capacity/IDs passed; E23 integer oracle/count16 passed; E24 interpolation_error="<<error<<" passed; extreme interpolation passed\n";
}
