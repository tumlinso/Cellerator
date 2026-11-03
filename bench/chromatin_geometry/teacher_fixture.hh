#pragma once
#include "device_fixture.hh"
#include "learning/teacher_chain.hh"
#include <cstring>
#include <vector>

namespace cgs {
// Controlled local teacher: one LUT per observed condition, initially a
// pass-through truth table, trained against eight explicit Boolean targets.
// The input predicate planes are produced separately by the supplied DAG.
struct TeacherFixture {
    using Node=teacher::Node; using Fit=teacher::FitSpan; using Stat=teacher::RowStatistics;
    std::vector<Node> hnodes; std::vector<std::uint32_t> hroots, levels;
    std::vector<Fit> fits;
    DeviceBuffer<Node> nodes;
    DeviceBuffer<std::uint32_t> roots, node_words, gate_words;
    DeviceBuffer<float> logits, targets, changes;
    DeviceBuffer<std::uint8_t> rows, luts, admission;
    DeviceBuffer<Stat> statistics;
    teacher::Program program; teacher::ChainReceipt receipt;
    std::uint64_t batch, conditions, words, teacher_generation=0;
    static std::vector<Node> make_nodes(std::uint64_t c){
        std::vector<Node> v(2*c);for(std::uint32_t i=0;i<c;++i){v[i].op=teacher::Op::predicate;v[i].x=i;v[c+i].op=teacher::Op::lut3;v[c+i].x=i;v[c+i].y=(i+1)%c;v[c+i].z=(i+2)%c;v[c+i].table=i;}return v;
    }
    static std::vector<std::uint32_t> make_roots(std::uint64_t c){std::vector<std::uint32_t> v(c);for(std::uint32_t i=0;i<c;++i)v[i]=c+i;return v;}
    static std::vector<float> make_logits(std::uint64_t c){std::vector<float> v(c*8);for(std::size_t i=0;i<v.size();++i)v[i]=(i%8&4)?2.f:-2.f;return v;}
    static std::vector<float> make_targets(std::uint64_t c){std::vector<float> v(c*8);for(std::size_t i=0;i<v.size();++i)v[i]=(i%8&4)?1.f:0.f;return v;}
    static std::vector<std::uint8_t> make_rows(std::uint64_t c){std::vector<std::uint8_t> v(c*8);for(std::size_t i=0;i<v.size();++i)v[i]=i%8;return v;}
    explicit TeacherFixture(std::uint64_t b,std::uint64_t c,bool allocate_admission=true):hnodes(make_nodes(c)),hroots(make_roots(c)),levels{0,static_cast<std::uint32_t>(c),static_cast<std::uint32_t>(2*c)},fits(c),
      nodes(hnodes),roots(hroots),node_words(2*c*((b+31)/32)),gate_words(c*((b+31)/32)),logits(make_logits(c)),targets(make_targets(c)),changes(c),rows(make_rows(c)),luts(c),admission(allocate_admission?b*c:0),statistics(c*8),batch(b),conditions(c),words((b+31)/32){
        for(std::uint64_t i=0;i<c;++i)fits[i]={8*i,8,0.01f};
    }
    GateView enqueue(const GateView& predicates,std::uint32_t* status,std::uint64_t parameter_revision,bool train,cudaStream_t stream,teacher::ChainEvents events={},bool make_bytes=true){
        auto before=predicates.revisions;before.teacher_revision=teacher_generation;before.parameter_revision=parameter_revision;
        auto after=before;if(train)++after.teacher_revision;
        if(!teacher::prepare_program(hnodes.data(),hnodes.size(),levels.data(),levels.size(),hroots.data(),hroots.size(),conditions,conditions,after,&program))throw std::runtime_error("invalid teacher fixture program");
        teacher::ChainSpec spec{};spec.program=&program;spec.fits=fits.data();spec.training_sample_count=8*conditions;spec.batch=batch;spec.table_count=conditions;spec.epochs=train?1:0;spec.before=before;spec.after=after;
        teacher::ChainBuffers b{};b.capacity={8*conditions,8*conditions,8*conditions,conditions,0,2*conditions,conditions,conditions*words,2*conditions*words,conditions*words,batch*conditions};
        b.logits=logits.ptr;b.training_rows=rows.ptr;b.training_targets=targets.ptr;b.statistics=statistics.ptr;b.luts=luts.ptr;b.table_mean_change=changes.ptr;
        if(make_bytes && !admission.ptr && batch*conditions)throw std::runtime_error("teacher admission storage not allocated");
        b.nodes=nodes.ptr;b.roots=roots.ptr;b.predicates=predicates.primal_words;b.node_words=node_words.ptr;b.gate_words=gate_words.ptr;b.admission=make_bytes?admission.ptr:nullptr;b.status=status;
        cuda_check(teacher::enqueue_chain(spec,b,events,&receipt,stream),"teacher chain");teacher_generation=after.teacher_revision;return receipt.gate;
    }
};
}
