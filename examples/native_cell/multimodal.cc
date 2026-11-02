#include <Cellerator/cell.hh>
#include <Cellerator/cells/access.hh>
#include <Cellerator/cells/event.hh>
#include <Cellerator/cells/expression.hh>

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <utility>

namespace {

using Token = std::uint32_t;
using EventView = cellerator::cells::EventView<Token>;
using ExpressionView = cellerator::cells::ExpressionView<double, std::string>;
using MutableExpressionView =
    cellerator::cells::MutableExpressionView<double, std::string>;

// An application-defined typed capability. Cell and the biological adapters
// do not need to know what this quality code means.
struct QualityView {
    int evidence_code;
};

// A Cell instance can compose several adapters and application-owned metadata
// while keeping each readout's type explicit.
class MultiModalInstance {
public:
    MultiModalInstance(
        const cellerator::memory::const_array_view<Token>& events,
        cellerator::memory::matrix_view<double>& expression,
        std::span<const std::string> feature_ids,
        std::string identity,
        int evidence_code)
        : events_(events, 0, events.count, identity),
          expression_(expression, feature_ids, 1, identity),
          identity_(std::move(identity)), evidence_code_(evidence_code) {}

    EventView read(cellerator::ViewTag<EventView>) const & {
        return events_.read(cellerator::ViewTag<EventView>{});
    }

    ExpressionView read(cellerator::ViewTag<ExpressionView>) const & {
        return expression_.read(cellerator::ViewTag<ExpressionView>{});
    }

    cellerator::CellIdentityView<std::string> read(
        cellerator::ViewTag<cellerator::CellIdentityView<std::string>>) const & {
        return {identity_};
    }

    QualityView read(cellerator::ViewTag<QualityView>) const & {
        return {evidence_code_};
    }

    MutableExpressionView write(
        cellerator::ViewTag<MutableExpressionView>) & {
        return expression_.write(cellerator::ViewTag<MutableExpressionView>{});
    }

private:
    cellerator::cells::EventPrefix<Token, std::string> events_;
    cellerator::cells::MutableDenseExpressionRow<
        double, std::string, std::string> expression_;
    std::string identity_;
    int evidence_code_;
};

// A model may be non-callable when this Cell is used for typed access only.
struct ViewsOnlyModel {};

} // namespace

int main() {
    std::array<Token, 4> event_storage{4, 7, 7, 9};
    cellerator::memory::const_array_view<Token> event_descriptor{
        event_storage.data(), event_storage.size(),
        {cellerator::memory::domain::host, -1, -1, 0}};

    // Two rows, three features, and one padding value per row.
    std::array<double, 8> expression_storage{
        1.0, 0.5, 2.0, -1.0,
        3.0, 4.5, 6.0, -2.0};
    cellerator::memory::matrix_view<double> expression_descriptor{
        expression_storage.data(), 2, 3, 4,
        {cellerator::memory::domain::host, -1, -1, 0}};
    const std::array<std::string, 3> feature_ids{
        "gene-A", "gene-B", "gene-C"};

    ViewsOnlyModel model;
    auto multimodal = cellerator::make_cell(
        model, MultiModalInstance(event_descriptor, expression_descriptor,
                                  std::span<const std::string>(feature_ids),
                                  "cell-M", 8));
    static_assert(decltype(multimodal)::supports<EventView>(),
                  "event readout should be a typed Cell capability");
    static_assert(decltype(multimodal)::supports<ExpressionView>(),
                  "expression readout should be a typed Cell capability");
    static_assert(decltype(multimodal)::supports_edit<MutableExpressionView>(),
                  "expression mutation should remain an explicit typed capability");

    const auto events = multimodal.view<EventView>().tokens;
    const auto expression = multimodal.view<ExpressionView>();
    const auto identity = multimodal.view<
        cellerator::CellIdentityView<std::string>>();
    const auto quality = multimodal.view<QualityView>();

    std::cout << identity.value << " events=" << events.count
              << " first=" << events.data[0]
              << " expression[" << expression.feature_ids[1] << "]="
              << expression.values.data[1]
              << " evidence=" << quality.evidence_code << '\n';

    // Mutation is explicit and limited to numeric values. The matrix owner
    // remains responsible for synchronization and value-generation tracking.
    auto editable = multimodal.edit<MutableExpressionView>();
    editable.values.data[1] = 5.25;
    const auto updated = multimodal.view<ExpressionView>();
    std::cout << "updated " << updated.feature_ids[1] << '='
              << updated.values.data[1] << '\n';

    // Legacy/runtime consumers can opt into a bridge for a selected set of
    // views. The typed Cell remains the source of the capabilities.
    auto bridge = cellerator::make_cell_views<
        cellerator::CellAccess<EventView>,
        cellerator::CellAccess<QualityView>,
        cellerator::CellEdit<MutableExpressionView>>(multimodal);
    cellerator::CellViews& runtime_views = bridge;
    if (!runtime_views.supports<EventView>() ||
        runtime_views.view<QualityView>().evidence_code != 8) {
        std::cerr << "selected runtime bridge did not expose its typed contracts\n";
        return 1;
    }
    auto bridge_edit = runtime_views.edit<MutableExpressionView>();
    bridge_edit.values.data[2] = 6.5;
    if (multimodal.view<ExpressionView>().values.data[2] != 6.5) {
        std::cerr << "runtime bridge edit did not reach its typed Cell source\n";
        return 1;
    }
    std::cout << "optional bridge: event-count="
              << runtime_views.view<EventView>().tokens.count
              << " edited=" << multimodal.view<ExpressionView>().values.data[2]
              << '\n';
}
