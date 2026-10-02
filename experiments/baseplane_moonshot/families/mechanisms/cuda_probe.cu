#include <cuda_runtime.h>
// Source-independent FP64 representatives. Caller owns device buffers, capacity and stream.
// Port path specializes one port/one interior; rejects a singular interior via validity byte.
__global__ void ce_moon_port_one(const double* blocks,const double* loads,double* schur,double* rhs,
                                unsigned char* valid,int count) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
  const double* a=blocks+4*k; const double* b=loads+2*k;
  bool ok=isfinite(a[0])&&isfinite(a[1])&&isfinite(a[2])&&isfinite(a[3])&&
          isfinite(b[0])&&isfinite(b[1])&&fabs(a[3])>1e-12;
  valid[k]=ok;
  if(ok){schur[k]=a[0]-a[1]*a[2]/a[3];rhs[k]=b[0]-a[1]*b[1]/a[3];}
}
// Dense P*d correction, d is a caller-provided coarse solve. No factorization/WMMA claim.
__global__ void ce_moon_coarse_apply(const double* prolongation,const double* delta,const double* x,
                                   double* output,int fine,int coarse,double alpha) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=fine)return;
  double update=0;for(int j=0;j<coarse;++j)update+=prolongation[k*coarse+j]*delta[j];
  output[k]=x[k]+alpha*update;
}
// Score only caller-selected compatible tuple values [candidate,role].
__global__ void ce_moon_factor_score(const double* values,const double* weights,double* scores,int count) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
  scores[k]=weights[0]+weights[1]*values[3*k]+weights[2]*values[3*k+1]+weights[3]*values[3*k+2];
}
// One thread per world over the same immutable topological numerical graph.
__global__ void ce_moon_world_affine(const int* lhs,const int* rhs,const double* left_weight,
                                    const double* right_weight,const double* bias,const unsigned char* override_mask,
                                    const double* override_value,double* states,int nodes,int worlds) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;if(w>=worlds)return;
  double* state=states+w*nodes;
  for(int n=0;n<nodes;++n)state[n]=override_mask[w*nodes+n]?override_value[w*nodes+n]:
    bias[n]+(lhs[n]<0?0:left_weight[n]*state[lhs[n]])+(rhs[n]<0?0:right_weight[n]*state[rhs[n]]);
}
