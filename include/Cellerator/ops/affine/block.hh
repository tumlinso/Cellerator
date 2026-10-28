#pragma once
#include <Cellerator/ops/numeric/weighted_relations.hh>
namespace ce_moon::effects {
template<std::size_t Blocks,std::size_t Width> struct BlockAffine {
    static_assert(Blocks>0&&Width>0,"nonempty block dimensions");
    std::array<unsigned,Blocks> p{};
    std::array<std::array<std::array<double,Width>,Width>,Blocks> matrix{};
    std::array<std::array<double,Width>,Blocks> bias{};
    static BlockAffine identity() {
        BlockAffine out;
        for(std::size_t b=0;b<Blocks;++b) {out.p[b]=b;for(std::size_t i=0;i<Width;++i)out.matrix[b][i][i]=1.;}
        return out;
    }
    void validate() const {
        std::array<bool,Blocks> seen{};
        for(std::size_t b=0;b<Blocks;++b) {
            if(p[b]>=Blocks||seen[p[b]])throw std::invalid_argument("invalid block permutation");
            seen[p[b]]=true;
            for(std::size_t i=0;i<Width;++i) {
                if(!std::isfinite(bias[b][i]))throw std::invalid_argument("nonfinite block bias");
                for(double v:matrix[b][i])if(!std::isfinite(v))throw std::invalid_argument("nonfinite block matrix");
            }
        }
    }
};
template<std::size_t B,std::size_t K> BlockAffine<B,K> compose(const BlockAffine<B,K>& left,const BlockAffine<B,K>& right) {
    left.validate();right.validate();BlockAffine<B,K> out;
    for(std::size_t b=0;b<B;++b) {
        const auto source=right.p[b];out.p[b]=left.p[source];
        for(std::size_t i=0;i<K;++i) {
            out.bias[b][i]=right.bias[b][i];
            for(std::size_t k=0;k<K;++k) out.bias[b][i]=finite(out.bias[b][i]+finite(right.matrix[b][i][k]*left.bias[source][k]));
            for(std::size_t j=0;j<K;++j) for(std::size_t k=0;k<K;++k)
                out.matrix[b][i][j]=finite(out.matrix[b][i][j]+finite(right.matrix[b][i][k]*left.matrix[source][k][j]));
        }
    }
    return out;
}
template<std::size_t B,std::size_t K> std::array<double,B*K> apply(const BlockAffine<B,K>& op,const std::array<double,B*K>& x) {
    op.validate();for(double v:x)if(!std::isfinite(v))throw std::invalid_argument("nonfinite block input");
    std::array<double,B*K> out{};
    for(std::size_t b=0;b<B;++b)for(std::size_t i=0;i<K;++i) {
        double v=op.bias[b][i];
        for(std::size_t j=0;j<K;++j)v=finite(v+finite(op.matrix[b][i][j]*x[op.p[b]*K+j]));
        out[b*K+i]=v;
    }
    return out;
}

} // namespace ce_moon::effects
