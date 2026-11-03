#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/cells/event.hh>
#include <Cellerator/cells/expression.hh>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using cellerator::memory::domain;
using cellerator::memory::placement;

template<class C, class V>
concept HasPublicRead = requires(const C& cell) {
    cell.read(cellerator::ViewTag<V>{});
};

template<class C, class V>
concept HasPublicWrite = requires(C& cell) {
    cell.write(cellerator::ViewTag<V>{});
};

template<class C, class V>
concept HasRvalueView = requires(C&& cell) {
    std::move(cell).template view<V>();
};

template<class C, class V>
concept HasRvalueEdit = requires(C&& cell) {
    std::move(cell).template edit<V>();
};

template<class C, class V>
concept HasRvalueRead = requires(C&& cell) {
    std::move(cell).read(cellerator::ViewTag<V>{});
};

template<class C, class V>
concept HasRvalueWrite = requires(C&& cell) {
    std::move(cell).write(cellerator::ViewTag<V>{});
};

template<class C>
concept HasRvalueBufferDescriptor = requires(C&& cell) {
    std::move(cell).buffer_descriptor();
};

using Event = cellerator::cells::EventPrefix<std::uint32_t, int>;
using EventTokens = cellerator::cells::EventView<std::uint32_t>;
using Dense = cellerator::cells::DenseExpressionRow<float, int, int>;
using MutableDense = cellerator::cells::MutableDenseExpressionRow<float, int, int>;
using Expression = cellerator::cells::ExpressionView<float, int>;
using MutableExpression = cellerator::cells::MutableExpressionView<float, int>;
using Identity = cellerator::CellIdentityView<int>;

static_assert(HasPublicRead<Event, EventTokens>);
static_assert(HasPublicRead<Event, Identity>);
static_assert(HasPublicRead<Dense, Expression>);
static_assert(HasPublicRead<Dense, Identity>);
static_assert(HasPublicRead<MutableDense, Expression>);
static_assert(HasPublicRead<MutableDense, Identity>);
static_assert(HasPublicWrite<MutableDense, MutableExpression>);
static_assert(!HasRvalueView<Event, EventTokens>);
static_assert(!HasRvalueBufferDescriptor<Event>);
static_assert(!HasRvalueView<Dense, Expression>);
static_assert(!HasRvalueView<MutableDense, Expression>);
static_assert(!HasRvalueEdit<MutableDense, MutableExpression>);
static_assert(!HasRvalueRead<Event, EventTokens>);
static_assert(!HasRvalueRead<Dense, Expression>);
static_assert(!HasRvalueWrite<MutableDense, MutableExpression>);
static_assert(!std::is_polymorphic_v<Event>);
static_assert(!std::is_polymorphic_v<Dense>);
static_assert(!std::is_polymorphic_v<MutableDense>);

template<class F>
bool throws_any(F&& function) {
    try {
        function();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

void test_event_prefix() {
    using cellerator::cells::EventPrefix;
    using cellerator::cells::EventView;

    std::vector<std::uint32_t> storage{11, 11, 19, 23, 23, 31};
    cellerator::memory::const_array_view<std::uint32_t> descriptor{
        storage.data(), storage.size(), {domain::host, -1, -1, 0}};

    EventPrefix<std::uint32_t, std::string> cell(descriptor, 1, 4, "lineage-X");
    assert(cell.offset() == 1 && cell.limit() == 4);
    assert(&cell.buffer_descriptor() == &descriptor);
    assert(cell.identity() == "lineage-X");
    assert(cell.supports<EventView<std::uint32_t>>());
    assert(cell.supports<cellerator::CellIdentityView<std::string>>());

    const auto events = cell.view<EventView<std::uint32_t>>();
    assert(events.tokens.count == 4);
    assert(events.tokens.data == storage.data() + 1);
    assert(events.tokens.data[0] == 11);
    assert(events.tokens.data[1] == 19);
    assert(events.tokens.data[2] == 23);
    assert(events.tokens.data[3] == 23);
    const auto identity = cell.view<cellerator::CellIdentityView<std::string>>();
    assert(identity.value == "lineage-X");

    // Repeated occurrences are preserved, not deduplicated.
    EventPrefix<std::uint32_t, int> repeated(descriptor, 0, 2, 7);
    const auto repeated_view = repeated.view<EventView<std::uint32_t>>();
    assert(repeated_view.tokens.count == 2);
    assert(repeated_view.tokens.data[0] == repeated_view.tokens.data[1]);

    EventPrefix<std::uint32_t, int> empty(descriptor, storage.size(), 0, 0);
    const auto empty_view = empty.view<EventView<std::uint32_t>>();
    assert(empty_view.tokens.count == 0);
    assert(empty_view.tokens.data == storage.data() + storage.size());

    assert(throws_any([&] {
        EventPrefix<std::uint32_t, int> invalid(descriptor, 7, 0, 0);
        (void)invalid;
    }));
    assert(throws_any([&] {
        EventPrefix<std::uint32_t, int> invalid(descriptor, 4, 3, 0);
        (void)invalid;
    }));
    if (std::numeric_limits<std::size_t>::max()
        > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        const cellerator::memory::const_array_view<std::uint32_t> huge_descriptor{
            storage.data(), std::numeric_limits<std::size_t>::max(), descriptor.where};
        const auto unrepresentable_offset = static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max()) + 1;
        assert(throws_any([&] {
            EventPrefix<std::uint32_t, int> invalid(
                huge_descriptor, unrepresentable_offset, 0, 0);
            (void)invalid;
        }));
    }

    // The descriptor object is stable while its pointer is refreshed after
    // the backing vector reallocates. Old handles retain their prefix bounds.
    const auto old_view = cell.view<EventView<std::uint32_t>>();
    const auto original_limit = cell.limit();
    storage.reserve(storage.capacity() + 32);
    storage.push_back(47);
    descriptor.data = storage.data();
    descriptor.count = storage.size();
    const auto refreshed_view = cell.view<EventView<std::uint32_t>>();
    assert(refreshed_view.tokens.data == storage.data() + 1);
    assert(refreshed_view.tokens.count == original_limit);
    assert(refreshed_view.tokens.data[0] == 11);
    assert(refreshed_view.tokens.data[3] == 23);
    (void)old_view; // It was invalidated by reallocation; intentionally unused.
    descriptor.count = cell.offset() + cell.limit() - 1;
    assert(throws_any([&] { (void)cell.view<EventView<std::uint32_t>>(); }));
    descriptor.count = storage.size();

    // Extending a prefix is explicit and leaves the earlier cell boundary intact.
    EventPrefix<std::uint32_t, std::string> extended(
        descriptor, cell.offset(), cell.limit() + 1, cell.identity());
    assert(cell.limit() == original_limit);
    assert(extended.limit() == original_limit + 1);
    assert(extended.view<EventView<std::uint32_t>>().tokens.data[4] == 31);

    std::vector<std::uint32_t> reordered{101, 103, 107, 109};
    cellerator::memory::const_array_view<std::uint32_t> reordered_descriptor{
        reordered.data(), reordered.size(), descriptor.where};
    EventPrefix<std::uint32_t, std::string> explicitly_rebound(
        descriptor, 0, 2, "lineage-X");
    explicitly_rebound.rebind(reordered_descriptor, 2, 2);
    assert(&explicitly_rebound.buffer_descriptor() == &reordered_descriptor);
    const auto rebound_view = explicitly_rebound.view<EventView<std::uint32_t>>();
    assert(explicitly_rebound.offset() == 2 && explicitly_rebound.limit() == 2);
    assert(rebound_view.tokens.data == reordered.data() + 2);
    assert(rebound_view.tokens.data[0] == 107 && rebound_view.tokens.data[1] == 109);
    // A failed rebind leaves the prior selection and descriptor in place.
    assert(throws_any([&] { explicitly_rebound.rebind(reordered_descriptor, 4, 1); }));
    assert(&explicitly_rebound.buffer_descriptor() == &reordered_descriptor);
    const auto retained_view = explicitly_rebound.view<EventView<std::uint32_t>>();
    assert(explicitly_rebound.offset() == 2 && explicitly_rebound.limit() == 2);
    assert(retained_view.tokens.data == reordered.data() + 2);

    struct NonCallableModel {} model;
    auto bound = cellerator::make_cell(
        model, EventPrefix<std::uint32_t, std::string>(descriptor, 1, 4, "lineage-X"));
    static_assert(decltype(bound)::template supports<EventView<std::uint32_t>>());
    assert(bound.view<EventView<std::uint32_t>>().tokens.data == storage.data() + 1);
    assert(bound.view<cellerator::CellIdentityView<std::string>>().value == "lineage-X");
}

void test_expression_views() {
    using cellerator::cells::DenseExpressionRow;
    using cellerator::cells::ExpressionView;
    using cellerator::cells::MutableDenseExpressionRow;
    using cellerator::cells::MutableExpressionView;

    constexpr std::size_t columns = 3;
    constexpr std::size_t stride = 5;
    std::vector<double> values{
        1.0, 2.0, 3.0, -90.0, -91.0,
        4.0, 5.0, 6.0, -92.0, -93.0};
    std::vector<std::string> features{"gene/a", "gene/b", "gene/c"};
    const std::span<const std::string> feature_span(features);

    cellerator::memory::matrix_view<double> mutable_descriptor{
        values.data(), 2, columns, stride, {domain::host, -1, -1, 0}};
    cellerator::memory::const_matrix_view<double> const_descriptor{
        values.data(), 2, columns, stride, mutable_descriptor.where};

    DenseExpressionRow<double, std::string, std::string> row(
        const_descriptor, feature_span, 1, "cell-B");
    assert(row.row() == 1);
    assert(row.identity() == "cell-B");
    assert((row.supports<ExpressionView<double, std::string>>()));
    assert((!row.supports_edit<MutableExpressionView<double, std::string>>()));
    assert(row.supports<cellerator::CellIdentityView<std::string>>());
    const auto read = row.view<ExpressionView<double, std::string>>();
    assert(read.values.count == columns);
    assert(read.values.data == values.data() + stride);
    assert(read.values.data[0] == 4.0 && read.values.data[2] == 6.0);
    assert(read.feature_ids.size() == features.size());
    assert(read.feature_ids[0] == "gene/a" && read.feature_ids[2] == "gene/c");
    static_assert(std::is_const_v<std::remove_reference_t<decltype(read.values.data[0])>>,
        "read view must expose const values");
    static_assert(std::is_const_v<typename decltype(read.feature_ids)::element_type>,
        "feature IDs must remain read-only");

    MutableDenseExpressionRow<double, std::string, int> mutable_row(
        mutable_descriptor, feature_span, 0, 42);
    assert((mutable_row.supports<ExpressionView<double, std::string>>()));
    assert((mutable_row.supports_edit<MutableExpressionView<double, std::string>>()));
    const auto mutable_identity = mutable_row.view<cellerator::CellIdentityView<int>>();
    assert(mutable_identity.value == 42);
    auto edit = mutable_row.edit<MutableExpressionView<double, std::string>>();
    assert(edit.values.count == columns);
    assert(edit.values.data == values.data());
    edit.values.data[1] = 123.5;
    assert(values[1] == 123.5);
    assert(edit.feature_ids[1] == "gene/b");

    struct ExpressionModel {} model;
    auto bound = cellerator::make_cell(
        model, MutableDenseExpressionRow<double, std::string, int>(
                   mutable_descriptor, feature_span, 1, 42));
    static_assert(decltype(bound)::template supports<ExpressionView<double, std::string>>());
    static_assert(decltype(bound)::template supports_edit<
                  MutableExpressionView<double, std::string>>());
    auto bound_edit = bound.edit<MutableExpressionView<double, std::string>>();
    bound_edit.values.data[0] = 456.0;
    assert(values[stride] == 456.0);

    // Invalid row, feature alignment, row stride, null storage for nonempty
    // shapes, and arithmetic overflow all fail during construction.
    assert(throws_any([&] {
        DenseExpressionRow<double, std::string, int> invalid(
            const_descriptor, feature_span, 2, 0);
        (void)invalid;
    }));
    std::vector<std::string> misaligned{"a", "b"};
    assert(throws_any([&] {
        DenseExpressionRow<double, std::string, int> invalid(
            const_descriptor, std::span<const std::string>(misaligned), 0, 0);
        (void)invalid;
    }));
    auto short_stride = const_descriptor;
    short_stride.row_stride = 2;
    assert(throws_any([&] {
        DenseExpressionRow<double, std::string, int> invalid(
            short_stride, feature_span, 0, 0);
        (void)invalid;
    }));
    auto null_data = const_descriptor;
    null_data.data = nullptr;
    assert(throws_any([&] {
        DenseExpressionRow<double, std::string, int> invalid(
            null_data, feature_span, 0, 0);
        (void)invalid;
    }));
    auto overflow = const_descriptor;
    overflow.rows = std::numeric_limits<std::size_t>::max();
    overflow.row_stride = std::numeric_limits<std::size_t>::max();
    assert(throws_any([&] {
        DenseExpressionRow<double, std::string, int> invalid(
            overflow, feature_span, 0, 0);
        (void)invalid;
    }));
    if (std::numeric_limits<std::size_t>::max()
        > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        auto unrepresentable_extent = const_descriptor;
        unrepresentable_extent.rows = static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max()) + 2;
        unrepresentable_extent.columns = 1;
        unrepresentable_extent.row_stride = 1;
        const std::span<const std::string> one_feature(features.data(), 1);
        assert(throws_any([&] {
            DenseExpressionRow<double, std::string, int> invalid(
                unrepresentable_extent, one_feature, 0, 0);
            (void)invalid;
        }));
    }

    // Selection validity is checked again after the borrowed descriptor changes.
    const_descriptor.row_stride = 2;
    assert(throws_any([&] { (void)row.view<ExpressionView<double, std::string>>(); }));
    const_descriptor.row_stride = stride;
    assert((row.view<ExpressionView<double, std::string>>().values.data
            == values.data() + stride));
}

void test_device_metadata_only() {
    using cellerator::cells::DenseExpressionRow;
    using cellerator::cells::EventPrefix;
    using cellerator::cells::EventView;
    using cellerator::cells::ExpressionView;

    // Use a valid host allocation with device placement metadata. The test
    // only checks descriptors and never reads through either returned pointer.
    std::uint32_t event_words[4]{};
    const placement device{domain::device, 3, -1, 0x5u};
    cellerator::memory::const_array_view<std::uint32_t> event_descriptor{
        event_words, 4, device};
    EventPrefix<std::uint32_t, int> events(event_descriptor, 1, 2, 99);
    const auto event_view = events.view<EventView<std::uint32_t>>();
    assert(event_view.tokens.data == event_words + 1);
    assert(event_view.tokens.where == device);

    double matrix_words[8]{};
    std::uint32_t feature_ids[2]{};
    cellerator::memory::const_matrix_view<double> matrix_descriptor{
        matrix_words, 2, 2, 4, device};
    DenseExpressionRow<double, std::uint32_t, int> expression(
        matrix_descriptor, std::span<const std::uint32_t>(feature_ids, 2),
        1, 7);
    const auto expression_view = expression.view<ExpressionView<double, std::uint32_t>>();
    assert(expression_view.values.data == matrix_words + 4);
    assert(expression_view.values.where == device);
    assert(expression_view.feature_ids.data() == feature_ids);
}

} // namespace

int main() {
    test_event_prefix();
    test_expression_views();
    test_device_metadata_only();
}
