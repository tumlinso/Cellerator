#include <Cellerator/nn/attention.hh>
#include <Cellerator/biology/coordinates.hh>
#include <ATen/Functions.h>
#include <torch/csrc/autograd/autograd.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace nn = cellerator::nn;
namespace bio = cellerator::biology;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    nn::Parameters parameters(at::Device(at::kCPU), at::kFloat);
    auto group = parameters.group();
    nn::RelationAttention attention(group, {.width=4, .heads=2});
    auto x = at::randn({2, 3, 4}, group.options()).set_requires_grad(true);
    const std::array<std::int64_t, 3> boundaries{0, 3, 6};
    auto order = bio::OccurrenceCoordinates<std::int64_t>::from_ids(
        {1, 2, 1, 3, 2, 1}, {0, 3, 6});
    nn::CsrSupport support;
    support.row_ptr.push_back(0);
    for (std::int64_t target=0; target<6; ++target) {
        for (std::int64_t source=(target/3)*3; source<(target/3)*3+3; ++source)
            support.col_indices.push_back(source);
        support.row_ptr.push_back(static_cast<std::int64_t>(support.col_indices.size()));
    }
    auto relation = nn::PreparedRelation::from_csr(support, order.order_token(), order.order_token());
    auto dense = attention(x);
    auto sparse = attention(x, relation);
    require(at::allclose(dense, sparse, 1e-5, 1e-6), "dense/CSR attention mismatch");
    auto packed = attention(x.reshape({6, 4}), nn::AttentionLayout{boundaries, boundaries});
    require(at::allclose(dense.reshape({6, 4}), packed, 1e-5, 1e-6), "packed grouping mismatch");
    auto g1 = torch::autograd::grad({dense.square().sum()}, {x, attention.parameters().query.weight});
    auto g2 = torch::autograd::grad({sparse.square().sum()}, {x, attention.parameters().query.weight});
    require(at::allclose(g1[0], g2[0], 1e-4, 1e-5), "dense/CSR input VJP mismatch");
    require(at::allclose(g1[1], g2[1], 1e-4, 1e-5), "dense/CSR weight VJP mismatch");

    nn::RelationAttention causal(attention.parameters(), {.width=4,.heads=2,.causal=true});
    auto original = x.detach();
    auto changed = original.clone();
    changed.select(0,0).select(0,2).add_(1000);
    changed.select(0,1).add_(2000);
    auto before = causal(original);
    auto after = causal(changed);
    require(at::allclose(before.select(0,0).narrow(0,0,2), after.select(0,0).narrow(0,0,2)),
            "future/adjacent rows leaked into causal context");
    auto filtered = relation.causal();
    require(!filtered.has_future_edges(), "causal preparation retained future edge");
    require(at::allclose(causal(original,filtered), before, 1e-5, 1e-6),
            "causal CSR/dense mismatch");

    nn::CsrSupport empty{{0,0,0,0,0,0,0},{}};
    auto empty_relation = nn::PreparedRelation::from_csr(empty,order.order_token(),order.order_token());
    auto empty_output = attention(original,empty_relation);
    require(empty_output.eq(0).all().item<bool>(), "empty neighborhoods must produce zero contribution");
    nn::CsrSupport duplicates{{0,2,3,3,3,3,3},{0,0,1}};
    auto duplicate_relation = nn::PreparedRelation::from_csr(duplicates,order.order_token(),order.order_token());
    require(attention(original,duplicate_relation).isfinite().all().item<bool>(), "duplicate support failed");

    auto vacant = at::empty({0,4},group.options());
    const std::array<std::int64_t,3> vacant_groups{0,0,0};
    require(attention(vacant,nn::AttentionLayout{vacant_groups,vacant_groups}).sizes()==vacant.sizes(),
            "empty packed groups shape mismatch");
    auto dense_empty = at::empty({2,0,4},group.options());
    require(attention(dense_empty).sizes()==dense_empty.sizes(), "empty dense shape mismatch");

    relation.col_indices().fill_(9999);
    bool rejected=false;
    try { attention(original,relation); } catch (const std::exception&) { rejected=true; }
    require(rejected, "mutated prepared structure reached attention");
    std::cout << "attention: dense/packed/CSR, VJPs, causal isolation, duplicates, empty groups, mutation passed\n";
}
