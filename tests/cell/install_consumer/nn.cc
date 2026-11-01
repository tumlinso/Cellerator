#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/nn.hh>

#include <cassert>
#include <memory>
#include <string>

int main() {
    namespace ce = cellerator;
    ce::nn::Parameters parameters(at::Device(at::kCPU), at::kFloat);
    assert(parameters.size() == 0);

    using Context = ce::modeling::StepContext<>;
    auto dynamics = [](const int& state, const Context&) { return state + 1; };
    auto model = ce::modeling::make_dynamical_model<int, Context>(dynamics);
    auto cell = ce::modeling::make_dynamical_cell(model, 3, std::string("cell-A"));
    const ce::Cell& base = cell;
    using Snapshot = ce::modeling::DynamicalStateSnapshotView<int>;
    auto before = base.view<Snapshot>();
    cell.advance();
    assert(*before.state == 3 && *base.view<Snapshot>().state == 4);
    assert(base.view<ce::CellIdentityView<std::string>>().value == "cell-A");
}
