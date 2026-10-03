#include <Cellerator/math/frontier/operators.hh>
#include <Cellerator/compute/operation/native_numeric/local_arithmetic.hh>
#include <algorithm>
#include <cfenv>
#include <cmath>
namespace cellerator::math::frontier {
namespace cm=ce_moon::mechanisms;
namespace nn=compute::native_numeric;
namespace {
struct rejected { status why; };
void require(bool condition,status why=status::invalid_binding) {if(!condition)throw rejected{why};}
void finite_matrix(const Matrix& a) {require(a.data.size()==Matrix::checked_size(a.rows,a.cols));cm::finite(a.data);}
void shape(const Matrix& a,std::uint64_t rows,std::uint64_t cols){finite_matrix(a);require(a.rows==rows&&a.cols==cols);}
void axis(const rel::axis_descriptor& a) {require(mx::valid_axis(a)&&a.extent,status::invalid_axes);}
bool same_axis(const rel::axis_descriptor& a,const rel::axis_descriptor& b){return a.extent==b.extent&&mx::nf::same_axis(a.identity,b.identity);}
void host_policy(){require(std::fegetround()==FE_TONEAREST,status::unsupported_policy);}
void generation(const generations* current){require(mx::validate_generation(current)==mx::status::success);}
void saved_generation(const generations* current,const generations& saved){generation(current);require(mx::same_generations(*current,saved),status::stale_generation);}
template<class Action> status protect(Action action) noexcept {
    try{host_policy();action();return status::success;}catch(rejected e){return e.why;}
    catch(const std::domain_error&){return status::ill_conditioned;}
    catch(const std::invalid_argument&){return status::invalid_binding;}
    catch(...){return status::provider_failure;}
}
Matrix add(const Matrix& a,const Matrix& b) {
    shape(b,a.rows,a.cols);finite_matrix(a);Matrix out(a.rows,a.cols);
    require(nn::local_forward(nn::local_operation::add,std::span<const double>(a.data),std::span<const double>(b.data),std::span<double>(out.data))==nn::local_status::success,status::provider_failure);
    cm::finite(out.data);return out;
}
Matrix subtract(const Matrix& a,const Matrix& b){auto negative=b;for(auto& value:negative.data)value=-value;return add(a,negative);}
Matrix transpose(const Matrix& a){finite_matrix(a);Matrix out(a.cols,a.rows);for(std::size_t i=0;i<a.rows;++i)for(std::size_t j=0;j<a.cols;++j)out(j,i)=a(i,j);return out;}
Matrix column(std::span<const double> a){return Matrix(a.size(),1,std::vector<double>(a.begin(),a.end()));}
long double norm(const Matrix& a){finite_matrix(a);long double result=0;for(std::size_t i=0;i<a.rows;++i){long double row=0;for(std::size_t j=0;j<a.cols;++j)row+=std::abs(a(i,j));result=std::max(result,row);}return result;}
void policy(solve_policy p){require(std::isfinite(p.pivot_relative)&&p.pivot_relative>0&&p.pivot_relative<1
    &&std::isfinite(p.residual_relative)&&p.residual_relative>0&&std::isfinite(p.max_rhs_amplification)&&p.max_rhs_amplification>=1,status::unsupported_policy);}
void native_composition_policy(solve_policy p){policy(p);require(p.pivot_relative>=1e-12,status::unsupported_policy);}
solve_diagnostics diagnose(const Matrix& a,const Matrix& x,const Matrix& b) {
    const auto residual=subtract(cm::multiply(a,x),b);auto an=norm(a),xn=norm(x),bn=norm(b);
    auto denominator=an*xn+bn;auto error=denominator?norm(residual)/denominator:norm(residual);
    auto amplification=bn?an*xn/bn:0;
    return {static_cast<double>(error),static_cast<double>(amplification)};
}
Matrix solve_checked(const Matrix& a,const Matrix& b,solve_policy p,solve_diagnostics& diagnostics){
    policy(p);shape(a,a.rows,a.rows);shape(b,a.rows,b.cols);require(a.rows&&b.cols);
    auto result=cm::solve(a,b,p.pivot_relative);diagnostics=diagnose(a,result,b);
    require(diagnostics.rhs_amplification<=p.max_rhs_amplification,status::ill_conditioned);
    require(diagnostics.normalized_residual<=p.residual_relative,status::residual_rejected);return result;
}
void reject_alias(const Matrix& output,std::initializer_list<const Matrix*> inputs){for(auto* input:inputs)require(&output!=input);}
void polynomial(square_axes axes,const polynomial_primal& p){axis(axes.rows);axis(axes.columns);generation(p.current);
    require(p.X&&p.L&&p.R&&p.M);auto n=axes.rows.extent,m=axes.columns.extent;
    shape(*p.X,n,m);shape(*p.L,n,n);shape(*p.R,m,m);shape(*p.M,m,n);
}
void polynomial(const polynomial_tape& t){polynomial(t.axes,t.primal);saved_generation(t.primal.current,t.saved);}
void multilevel(multilevel_descriptor d,const multilevel_primal& p){axis(d.fine);axis(d.coarse);generation(p.current);
    require(p.D&&p.P&&p.K&&p.R&&p.state);auto n=d.fine.extent,k=d.coarse.extent;
    shape(*p.D,n,n);shape(*p.P,n,k);shape(*p.K,k,k);shape(*p.R,k,n);require(p.state->size()==n);cm::finite(*p.state);
    require(d.meaning==label::exact_identity||d.meaning==label::model_restriction||d.meaning==label::approximation);
}
}
status polynomial_forward(square_axes axes,const polynomial_primal& p,polynomial_result& out) noexcept {return protect([&]{
    polynomial(axes,p);reject_alias(out.value,{p.X,p.L,p.R,p.M});auto value=add(add(cm::multiply(*p.L,*p.X),cm::multiply(*p.X,*p.R)),cm::multiply(cm::multiply(*p.X,*p.M),*p.X));
    polynomial_result next{std::move(value),{axes,p,*p.current},label::model_restriction};out=std::move(next);
});}
status polynomial_delta(const polynomial_tape& t,const Matrix& D,Matrix& out) noexcept {return protect([&]{
    polynomial(t);shape(D,t.axes.rows.extent,t.axes.columns.extent);const auto& p=t.primal;reject_alias(out,{p.X,p.L,p.R,p.M,&D});
    auto value=add(cm::multiply(*p.L,D),cm::multiply(D,*p.R));
    value=add(value,cm::multiply(cm::multiply(D,*p.M),*p.X));
    value=add(value,cm::multiply(cm::multiply(*p.X,*p.M),D));
    value=add(value,cm::multiply(cm::multiply(D,*p.M),D)); // Exact finite delta includes DMD.
    out=std::move(value);
});}
status polynomial_jvp(const polynomial_tape& t,const Matrix& dX,const Matrix& dL,const Matrix& dR,const Matrix& dM,Matrix& out) noexcept {return protect([&]{
    polynomial(t);const auto& p=t.primal;reject_alias(out,{p.X,p.L,p.R,p.M,&dX,&dL,&dR,&dM});shape(dX,p.X->rows,p.X->cols);shape(dL,p.L->rows,p.L->cols);shape(dR,p.R->rows,p.R->cols);shape(dM,p.M->rows,p.M->cols);
    auto value=add(add(cm::multiply(dL,*p.X),cm::multiply(*p.L,dX)),add(cm::multiply(dX,*p.R),cm::multiply(*p.X,dR)));
    value=add(value,cm::multiply(cm::multiply(dX,*p.M),*p.X));
    value=add(value,cm::multiply(cm::multiply(*p.X,dM),*p.X));
    value=add(value,cm::multiply(cm::multiply(*p.X,*p.M),dX));out=std::move(value);
});}
status polynomial_vjp(const polynomial_tape& t,const Matrix& G,polynomial_adjoints& out) noexcept {return protect([&]{
    polynomial(t);const auto& p=t.primal;for(const auto* output:{&out.X,&out.L,&out.R,&out.M})reject_alias(*output,{p.X,p.L,p.R,p.M,&G});shape(G,p.X->rows,p.X->cols);auto xt=transpose(*p.X);
    auto gx=add(cm::multiply(transpose(*p.L),G),cm::multiply(G,transpose(*p.R)));
    gx=add(gx,cm::multiply(G,transpose(cm::multiply(*p.M,*p.X))));
    gx=add(gx,cm::multiply(transpose(cm::multiply(*p.X,*p.M)),G));
    polynomial_adjoints next{std::move(gx),cm::multiply(G,xt),cm::multiply(xt,G),cm::multiply(cm::multiply(xt,G),xt)};out=std::move(next);
});}
status checked_solve(const solve_primal& p,solve_policy limits,solve_result& out) noexcept {return protect([&]{
    axis(p.coordinates);axis(p.right_hand_sides);generation(p.current);require(p.A&&p.B);
    reject_alias(out.value,{p.A,p.B});reject_alias(out.tape.solution,{p.A,p.B});
    shape(*p.A,p.coordinates.extent,p.coordinates.extent);shape(*p.B,p.coordinates.extent,p.right_hand_sides.extent);
    solve_result next;next.value=solve_checked(*p.A,*p.B,limits,next.diagnostics);
    next.tape={p,*p.current,next.value,limits};out=std::move(next);
});}
status solve_jvp(const solve_tape& t,const Matrix& dA,const Matrix& dB,solve_result& out) noexcept {return protect([&]{
    saved_generation(t.primal.current,t.saved);require(t.primal.A&&t.primal.B);
    reject_alias(out.value,{t.primal.A,t.primal.B,&t.solution,&dA,&dB});reject_alias(out.tape.solution,{t.primal.A,t.primal.B,&t.solution,&dA,&dB});axis(t.primal.coordinates);axis(t.primal.right_hand_sides);
    shape(*t.primal.A,t.primal.coordinates.extent,t.primal.coordinates.extent);shape(*t.primal.B,t.primal.coordinates.extent,t.primal.right_hand_sides.extent);
    shape(dA,t.primal.A->rows,t.primal.A->cols);shape(dB,t.primal.B->rows,t.primal.B->cols);
    shape(t.solution,t.primal.B->rows,t.primal.B->cols);
    auto rhs=subtract(dB,cm::multiply(dA,t.solution));solve_result next;
    next.value=solve_checked(*t.primal.A,rhs,t.policy,next.diagnostics);
    // A derivative result has no primal tape: recursive differentiation is unsupported.
    out=std::move(next);
});}
status solve_port_regions(std::span<const port_region> regions,solve_policy limits,port_solution& out) noexcept {return protect([&]{
    require(!regions.empty());native_composition_policy(limits);const auto ports=regions[0].boundary;axis(ports);
    std::vector<cm::PortResponse> responses;port_solution next;std::size_t total=ports.extent;
    for(std::size_t region_index=0;region_index<regions.size();++region_index){const auto& region=regions[region_index];
        axis(region.boundary);axis(region.interior);generation(region.current);
        require(!mx::nf::same_axis(region.interior.identity,ports.identity),status::invalid_axes);
        for(std::size_t prior=0;prior<region_index;++prior)
            require(!mx::nf::same_axis(region.interior.identity,regions[prior].interior.identity),status::invalid_axes);
        require(region.boundary_load!=&out.boundary&&region.interior_load!=&out.boundary);
        for(const auto& interior:out.interiors)require(region.boundary_load!=&interior&&region.interior_load!=&interior);
        require(same_axis(ports,region.boundary),status::invalid_axes);
        require(region.Abb&&region.Abi&&region.Aib&&region.Aii&&region.boundary_load&&region.interior_load);
        auto n=ports.extent,i=region.interior.extent;shape(*region.Abb,n,n);shape(*region.Abi,n,i);shape(*region.Aib,i,n);shape(*region.Aii,i,i);
        require(region.boundary_load->size()==n&&region.interior_load->size()==i);cm::finite(*region.boundary_load);cm::finite(*region.interior_load);
        // Admit the interior response and load under the same caller solve policy.
        solve_diagnostics diagnostic;solve_checked(*region.Aii,*region.Aib,limits,diagnostic);
        solve_checked(*region.Aii,column(*region.interior_load),limits,diagnostic);
        responses.push_back(cm::condense_ports(*region.Abb,*region.Abi,*region.Aib,*region.Aii,*region.boundary_load,*region.interior_load));
        next.borrowed_regions.push_back(region);next.saved.push_back(*region.current);require(i<=SIZE_MAX-total);total+=i;
    }
    auto system=cm::compose_port_system(responses);auto boundary=solve_checked(system.stiffness,column(system.load),limits,next.diagnostics);
    next.boundary=std::move(boundary.data);
    // Single-region composition retains the actual original solve_ports route.
    if(responses.size()==1){auto original=cm::solve_ports(responses[0]);require(original.size()==next.boundary.size());}
    Matrix full(total,total),rhs(total,1),solution(total,1);for(std::size_t j=0;j<ports.extent;++j)solution(j,0)=next.boundary[j];
    std::size_t offset=ports.extent;
    for(std::size_t r=0;r<regions.size();++r){const auto& region=regions[r];auto recovered=cm::reconstruct_interior(responses[r],next.boundary);
        next.interiors.push_back(recovered);
        for(std::size_t i=0;i<ports.extent;++i){rhs(i,0)+=(*region.boundary_load)[i];for(std::size_t j=0;j<ports.extent;++j)full(i,j)+=(*region.Abb)(i,j);
            for(std::size_t j=0;j<region.interior.extent;++j)full(i,offset+j)=(*region.Abi)(i,j);}
        for(std::size_t i=0;i<region.interior.extent;++i){rhs(offset+i,0)=(*region.interior_load)[i];solution(offset+i,0)=recovered[i];
            for(std::size_t j=0;j<ports.extent;++j)full(offset+i,j)=(*region.Aib)(i,j);
            for(std::size_t j=0;j<region.interior.extent;++j)full(offset+i,offset+j)=(*region.Aii)(i,j);}
        offset+=region.interior.extent;
    }
    next.full_normalized_residual=diagnose(full,solution,rhs).normalized_residual;
    require(next.full_normalized_residual<=limits.residual_relative,status::residual_rejected);out=std::move(next);
});}
bool port_solution_is_current(const port_solution& solution) noexcept {
    if(solution.borrowed_regions.size()!=solution.saved.size()||solution.saved.empty())return false;
    for(std::size_t i=0;i<solution.saved.size();++i){auto* live=solution.borrowed_regions[i].current;
        if(mx::validate_generation(live)!=mx::status::success||!mx::same_generations(*live,solution.saved[i]))return false;}
    return true;
}
status multilevel_apply(multilevel_descriptor descriptor,const multilevel_primal& p,multilevel_result& out) noexcept {return protect([&]{
    multilevel(descriptor,p);require(&out.value!=p.state);auto value=cm::matvec(*p.D,*p.state),coarse=cm::matvec(*p.P,cm::matvec(*p.K,cm::matvec(*p.R,*p.state)));
    for(std::size_t i=0;i<value.size();++i)value[i]+=coarse[i];
    cm::finite(value);
    multilevel_result next{std::move(value),{descriptor,p,*p.current},descriptor.meaning};out=std::move(next);
});}
status multilevel_input_vjp(const multilevel_tape& t,std::span<const double> G,std::vector<double>& out) noexcept {return protect([&]{
    multilevel(t.descriptor,t.primal);saved_generation(t.primal.current,t.saved);require(G.size()==t.descriptor.fine.extent);
    const auto& p=t.primal;require(&out!=p.state);std::vector<double> cotangent(G.begin(),G.end());cm::finite(cotangent);
    auto value=cm::matvec(transpose(*p.D),cotangent),coarse=cm::matvec(transpose(*p.R),cm::matvec(transpose(*p.K),cm::matvec(transpose(*p.P),cotangent)));
    for(std::size_t i=0;i<value.size();++i)value[i]+=coarse[i];
    cm::finite(value);out=std::move(value);
});}
status residual_correction(multilevel_descriptor descriptor,const Matrix& A,const Matrix& P,const Matrix& R,
    std::span<const double> x,std::span<const double> rhs,double alpha,double max_ratio,solve_policy limits,correction_result& out) noexcept {return protect([&]{
    native_composition_policy(limits);axis(descriptor.fine);axis(descriptor.coarse);auto n=descriptor.fine.extent,k=descriptor.coarse.extent;
    shape(A,n,n);shape(P,n,k);shape(R,k,n);require(x.size()==n&&rhs.size()==n);
    require(std::isfinite(alpha)&&std::isfinite(max_ratio)&&max_ratio>=0,status::unsupported_policy);
    std::vector<double> state(x.begin(),x.end()),load(rhs.begin(),rhs.end());cm::finite(state);cm::finite(load);
    auto residual=cm::residual(A,state,load);auto coarse=cm::multiply(cm::multiply(R,A),P);
    solve_diagnostics diagnostic;solve_checked(coarse,column(cm::matvec(R,residual)),limits,diagnostic);
    auto direction=cm::coarse_direction(A,P,R,state,load);auto value=state;
    for(std::size_t i=0;i<n;++i)value[i]+=alpha*direction[i];
    cm::finite(value);
    auto before=norm(column(residual)),after=norm(column(cm::residual(A,value,load)));
    require(after<=static_cast<long double>(max_ratio)*before,status::residual_rejected);
    correction_result next{std::move(value),static_cast<double>(before),static_cast<double>(after),label::approximation};out=std::move(next);
});}
} // namespace cellerator::math::frontier
