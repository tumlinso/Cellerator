#include <Cellerator/compute/operation/product2/product2.hh>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>
using namespace cellerator::compute::product2;
using namespace cellerator::execution::program;
namespace execution=cellerator::execution;
void require(bool v){if(!v)throw std::runtime_error("product2 required check failed");}
int main(){
 static_assert(!std::is_copy_constructible_v<prepared_owner> && !std::is_move_constructible_v<prepared_owner>);
 int64_t a[]={0,0,1,2},b[]={0,1,0,2};float x[]={0,2,3},k[]={2,3,4,5},y[]={-9,-9,-9,-9},g[]={1,1,1,1},dx[]={1,2,3},dk[]={1,1,1,1},dy[4]{},gx[3]{},gk[4]{};
 ce_product2_context*c=nullptr;require(ce_product2_create(3,4,a,4,b,4,8,&c)==0);
 ce_product2_binding v{};v.x=x;v.x_count=3;v.k=k;v.k_count=4;v.y=y;v.y_count=4;v.g=g;v.g_count=4;v.dx=dx;v.dx_count=3;v.dk=dk;v.dk_count=4;v.dy=dy;v.dy_count=4;v.gx=gx;v.gx_count=3;v.gk=gk;v.gk_count=4;v.expected_structure_generation=v.current_structure_generation=8;
 require(ce_product2_forward(c,&v)==0);require(y[0]==0 && y[1]==0 && y[2]==0 && y[3]==45);
 require(ce_product2_vjp(c,&v)==0);require(gx[0]==14 && gx[1]==0 && gx[2]==30 && gk[3]==9);
 require(ce_product2_jvp(c,&v)==0);require(dy[0]==0 && dy[1]==6 && dy[2]==8 && dy[3]==99);
 float lhs=0,rhs=0;for(int i=0;i<4;++i)lhs+=dy[i]*g[i];for(int i=0;i<3;++i)rhs+=dx[i]*gx[i];for(int i=0;i<4;++i)rhs+=dk[i]*gk[i];require(lhs==rhs);
 alignas(float) unsigned char misaligned[20]{};auto mis=v;mis.x=reinterpret_cast<const float*>(misaligned+1);require(ce_product2_forward(c,&mis)==CE_PRODUCT2_INVALID_ARGUMENT);
 ce_product2_context* overflowctx=nullptr;require(ce_product2_create(3,4,a,UINT64_MAX,b,4,8,&overflowctx)==CE_PRODUCT2_OVERFLOW && !overflowctx);
 alignas(int64_t) unsigned char misindex[40]{};ce_product2_context* misctx=nullptr;require(ce_product2_create(3,4,reinterpret_cast<const int64_t*>(misindex+1),4,b,4,8,&misctx)==CE_PRODUCT2_INVALID_ARGUMENT && !misctx);
 auto bad=v;bad.y=x;bad.y_count=3;require(ce_product2_forward(c,&bad)==CE_PRODUCT2_INVALID_ARGUMENT);bad=v;bad.gx=y;bad.gx_count=4;bad.gk=y;require(ce_product2_vjp(c,&bad)==CE_PRODUCT2_ALIAS);
 bad=v;bad.current_value_generation=1;y[0]=-42;require(ce_product2_forward(c,&bad)==CE_PRODUCT2_STALE_GENERATION && y[0]==-42);
 bad=v;bad.x_count=UINT64_MAX;require(ce_product2_forward(c,&bad)==CE_PRODUCT2_OVERFLOW && y[0]==-42);
 auto invalid=a[3];a[3]=3;ce_product2_context*reject=nullptr;require(ce_product2_create(3,4,a,4,b,4,8,&reject)==CE_PRODUCT2_INVALID_INDEX && !reject);a[3]=invalid;
 prepared_owner owner({3,4,8,a,b,4,4});bound_view good{v},late{bad};stage_state states[]={{&owner,operation::forward},{&owner,operation::forward}};auto first=make_stage(states[0],1,0),second=make_stage(states[1],2,1);prepared_stage_v2 stages[]={first,second};uint64_t dep=0;stages[1].dependency_count=1;prepared_program_v2 program{2,0,stages,2,&dep,1};launch_binding_v2 bindings[2]{};bindings[0].input=&good;bindings[1].input=&late;
 y[0]=-123;require(execute_prepared_program_v2(program,bindings,2,nullptr)!=program_status::success);require(y[0]==-123);
 late=good;require(execute_prepared_program_v2(program,bindings,2,nullptr)==program_status::success && y[3]==45);
 
 execution::persistent_axis_identity axis{{execution::biological_abi_version,execution::serialized_record_kind::persistent_axis_identity,sizeof(execution::persistent_axis_identity)},{1,1},{2,2},{3,3},{4,4}};
 descriptor axes{3,4,8,a,b,4,4,true,{axis,3},{axis,4},{axis,4}};prepared_owner typed(axes);bound_view typed_binding{v,axis,axis,axis};
 require(typed.execute(typed_binding,operation::forward)==CE_PRODUCT2_SUCCESS);
 require(typed.execute(v,operation::forward)==CE_PRODUCT2_INVALID_ARGUMENT);
 typed_binding.input_axis.order.low=88;y[0]=-999;require(typed.execute(typed_binding,operation::forward)==CE_PRODUCT2_INVALID_ARGUMENT && y[0]==-999);
 typed_binding={v,axis,axis,axis};typed_binding.product_axis.header.schema_version=0;require(typed.admit(typed_binding,operation::forward)==CE_PRODUCT2_INVALID_ARGUMENT);
 ce_product2_destroy(c);c=nullptr;require(ce_product2_create(0,0,nullptr,0,nullptr,0,0,&c)==0);ce_product2_binding empty{};require(ce_product2_forward(c,&empty)==0 && ce_product2_vjp(c,&empty)==0 && ce_product2_jvp(c,&empty)==0);ce_product2_destroy(c);
 std::cout<<"native product2 CPU direct/VJP/JVP/zero/repeated/admission/prepared-program PASS\n";
}
