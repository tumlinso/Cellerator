#pragma once
#include <ce_moon/reference.hpp>
namespace ce_moon::tensor {
using Matrix16=std::array<float,256>;
using Ids16=std::array<std::uint64_t,16>;
using PairMask=std::array<unsigned char,256>;
struct TileShape { unsigned rows=16,features=16,outputs=16; };
inline void validate(TileShape s){if(!s.rows||s.rows>16||!s.features||s.features>16||!s.outputs||s.outputs>16)throw std::invalid_argument("tile axes must be 1..16");}
inline void finite(float x){if(!std::isfinite(x))throw std::invalid_argument("nonfinite tensor value");}
// Rows are objects/states, contraction axis is features, columns are outputs.
// Inactive source elements are ignored; returned padding is explicitly zero.
inline Matrix16 multiply(const Matrix16& a,const Matrix16& b,TileShape s){
 validate(s);Matrix16 out{};
 for(unsigned i=0;i<s.rows;++i)for(unsigned j=0;j<s.outputs;++j){double v=0;
  for(unsigned k=0;k<s.features;++k){finite(a[i*16+k]);finite(b[k*16+j]);v+=double(a[i*16+k])*b[k*16+j];}
  out[i*16+j]=float(v);finite(out[i*16+j]);}
 return out;
}
struct ObjectTile { Matrix16 values{}; Ids16 ids{}; unsigned rows=0,outputs=0; };
inline ObjectTile object_features(const Matrix16& x,const Matrix16& weights,TileShape s,const Ids16& ids){
 ObjectTile out{multiply(x,weights,s),{},s.rows,s.outputs};std::copy_n(ids.begin(),s.rows,out.ids.begin());return out;
}
// Directed score(i,j)=sum_k Q(i,k)*K(j,k). Caller constructs role-specific Q,K.
inline Matrix16 relation_scores(const Matrix16& q,const Matrix16& k,unsigned rows,unsigned features){
 validate({rows,features,rows});Matrix16 kt{};
 for(unsigned j=0;j<rows;++j)for(unsigned f=0;f<features;++f)kt[f*16+j]=k[j*16+f];
 return multiply(q,kt,{rows,features,rows});
}
struct Pair {std::uint64_t from,to;unsigned row,column;float score;};
struct RelationBatch {std::vector<Pair> pairs;std::size_t required=0;bool overflow=false;};
// Mask encodes caller supplied self/strand/support policy; no source interpretation.
// Stable row-major emit truncates at capacity and counts every eligible pair.
inline RelationBatch compact_relations(const Matrix16& scores,const Ids16& ids,unsigned rows,const PairMask& mask,float threshold,std::size_t capacity){
 validate({rows,1,1});finite(threshold);RelationBatch out;
 for(unsigned i=0;i<rows;++i)for(unsigned j=0;j<rows;++j){auto n=i*16+j;finite(scores[n]);if(mask[n]>1)throw std::invalid_argument("nonbinary pair mask");
  if(mask[n]&&scores[n]>threshold){++out.required;if(out.pairs.size()<capacity)out.pairs.push_back({ids[i],ids[j],i,j,scores[n]});}}
 out.overflow=out.required>capacity;return out;
}
// Exactly one two-step binary product: each integer count is in [0,16].
inline Matrix16 finite_relation_counts(const Relation16& a,const Relation16& b){
 Matrix16 af{},bf{};for(unsigned i=0;i<16;++i)for(unsigned j=0;j<16;++j){if(a[i][j]>1||b[i][j]>1)throw std::invalid_argument("nonbinary relation");af[i*16+j]=a[i][j];bf[i*16+j]=b[i][j];}return multiply(af,bf,{});
}
inline Relation16 existence(const Matrix16& counts){Relation16 out{};for(unsigned i=0;i<16;++i)for(unsigned j=0;j<16;++j){auto x=counts[i*16+j];finite(x);if(x<0)throw std::invalid_argument("negative path count");out[i][j]=x>0;}return out;}
// Region-conditioned weights shared across candidate entry-state rows. Tanh
// makes interpolation a measurable approximation, even with an affine input map.
inline Matrix16 possible_states(const Matrix16& states,const Matrix16& weights,TileShape s){auto out=multiply(states,weights,s);for(unsigned i=0;i<s.rows;++i)for(unsigned j=0;j<s.outputs;++j)out[i*16+j]=std::tanh(out[i*16+j]);return out;}
struct Interpolated {std::array<float,16> values{};unsigned lower=0,upper=0;float alpha=0;};
// Scalar sampling coordinate is explicit. No extrapolation or multidimensional
// completeness is claimed; inactive coordinates are ignored.
inline Interpolated interpolate(const Matrix16& table,const std::array<float,16>& coordinates,unsigned rows,unsigned outputs,float query){
 validate({rows,1,outputs});finite(query);for(unsigned i=0;i<rows;++i){finite(coordinates[i]);if(i&&coordinates[i]<=coordinates[i-1])throw std::invalid_argument("coordinates not increasing");}
 if(query<coordinates[0]||query>coordinates[rows-1])throw std::out_of_range("unsampled query");
 Interpolated out;while(out.upper+1<rows&&coordinates[out.upper]<query)++out.upper;
 out.lower=out.upper?out.upper-1:0;if(coordinates[out.upper]==query)out.lower=out.upper;
 if(out.lower!=out.upper)out.alpha=(query-coordinates[out.lower])/(coordinates[out.upper]-coordinates[out.lower]);
 for(unsigned j=0;j<outputs;++j){float a=table[out.lower*16+j],b=table[out.upper*16+j];finite(a);finite(b);out.values[j]=a+(b-a)*out.alpha;}return out;
}
} // namespace ce_moon::tensor
