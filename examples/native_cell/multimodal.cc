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

// An application-defined capability. Cell does not need to know this type.
struct QualityView {
    int evidence_code;
};

class MultiModalCell final
    : public cellerator::CellAccess<EventView>,
      public cellerator::CellAccess<ExpressionView>,
      public cellerator::CellEdit<MutableExpressionView>,
      public cellerator::CellAccess<cellerator::CellIdentityView<std::string>>,
      public cellerator::CellAccess<QualityView> {
public:
    MultiModalCell(
        const cellerator::memory::const_array_view<Token>& events,
        cellerator::memory::matrix_view<double>& expression,
        std::span<const std::string> feature_ids,
        std::string identity,
        int evidence_code)
        : events_(events, 0, events.count, identity),
          expression_(expression, feature_ids, 1, identity),
          identity_(std::move(identity)), evidence_code_(evidence_code) {}

protected:
    EventView read(cellerator::ViewTag<EventView>) const & override {
        return events_.view<EventView>();
    }

    ExpressionView read(cellerator::ViewTag<ExpressionView>) const & override {
        return expression_.view<ExpressionView>();
    }

    cellerator::CellIdentityView<std::string> read(
        cellerator::ViewTag<cellerator::CellIdentityView<std::string>>) const & override {
        return {identity_};
    }

    QualityView read(cellerator::ViewTag<QualityView>) const & override {
        return {evidence_code_};
    }

    MutableExpressionView write(
        cellerator::ViewTag<MutableExpressionView>) & override {
        return expression_.edit<MutableExpressionView>();
    }

private:
    cellerator::cells::EventCell<Token, std::string> events_;
    cellerator::cells::MutableDenseExpressionCell<
        double, std::string, std::string> expression_;
    std::string identity_;
    int evidence_code_;
};

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

    MultiModalCell multimodal(event_descriptor, expression_descriptor,
                              std::span<const std::string>(feature_ids),
                              "cell-M", 8);
    const cellerator::Cell& cell = multimodal;

    const auto events = cell.view<EventView>().tokens;
    const auto expression = cell.view<ExpressionView>();
    const auto identity = cell.view<cellerator::CellIdentityView<std::string>>();
    const auto quality = cell.view<QualityView>();

    std::cout << identity.value << " events=" << events.count
              << " first=" << events.data[0]
              << " expression[" << expression.feature_ids[1] << "]="
              << expression.values.data[1]
              << " evidence=" << quality.evidence_code << '\n';

    // Mutation is explicit and limited to numeric values. The matrix owner
    // remains responsible for synchronization and generation tracking.
    auto editable = multimodal.edit<MutableExpressionView>();
    editable.values.data[1] = 5.25;
    const auto updated = cell.view<ExpressionView>();
    std::cout << "updated " << updated.feature_ids[1] << '='
              << updated.values.data[1] << '\n';
}
