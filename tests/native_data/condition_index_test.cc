#include <Cellerator/execution/dependency/condition_index.hh>
#include <Cellerator/ops/mechanism/gated_product_binding.hh>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dependency = cellerator::geometry::dependency;
namespace execution = cellerator::execution;
namespace mechanism = cellerator::ops::mechanism;
namespace structure = cellerator::structure;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Exception, class Function>
bool throws(Function&& function) {
    try {
        function();
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

std::vector<std::uint64_t> ids(std::span<const dependency::condition_id> values) {
    std::vector<std::uint64_t> result;
    result.reserve(values.size());
    for (const auto value : values) result.push_back(value.low);
    std::sort(result.begin(), result.end());
    return result;
}

void require_ids(std::span<const dependency::condition_id> actual,
                 std::initializer_list<std::uint64_t> expected,
                 const char* message) {
    auto found = ids(actual);
    std::vector<std::uint64_t> wanted(expected);
    std::sort(wanted.begin(), wanted.end());
    require(found == wanted, message);
}

dependency::SourceDescriptor source(std::uint64_t id, std::uint64_t domain,
                                    std::uint64_t order, std::uint64_t rep,
                                    std::uint64_t extent) {
    const structure::domain_id d{domain, 1};
    return {{dependency::source_id{id, 1}, d,
             dependency::representation_id{rep, 1}},
            {d, structure::order_id{order, 1}, extent}};
}

dependency::SelectionDescriptor all(std::uint64_t id,
                                    const dependency::SourceDescriptor& src) {
    return {{{id, 1}, src.axis.domain, dependency::SelectionSpace::coordinates},
            src.axis.logical_order, src.axis.extent};
}

dependency::SelectionDescriptor range(std::uint64_t id,
                                      const dependency::SourceDescriptor& src,
                                      std::uint64_t begin, std::uint64_t count) {
    auto result = all(id, src);
    result.kind = dependency::SelectionKind::range;
    result.begin = begin;
    result.count = count;
    return result;
}

struct Fixture {
    std::array<dependency::SourceDescriptor, 4> sources{
        source(1, 10, 20, 30, 8),  // ordinary state
        source(2, 10, 20, 31, 4),  // condition-only context
        source(3, 11, 21, 32, 3),  // ordinary parameter source
        source(4, 12, 22, 33, 0)   // valid zero-sized axis
    };

    std::array<dependency::SelectionDescriptor, 8> selections{
        range(101, sources[0], 0, 2),
        range(102, sources[0], 5, 2),
        range(103, sources[1], 1, 2),
        range(104, sources[1], 3, 1),
        range(105, sources[2], 0, 1),
        range(106, sources[2], 2, 1),
        range(107, sources[0], 8, 0),
        all(108, sources[3])
    };

    std::array<dependency::ReadDependency, 3> reads_a{{
        {sources[0].reference, selections[0].reference},
        {sources[1].reference, selections[2].reference},
        {sources[2].reference, selections[4].reference}
    }};
    std::array<dependency::ReadDependency, 3> reads_b{{
        {sources[0].reference, selections[1].reference},
        {sources[1].reference, selections[2].reference},
        {sources[2].reference, selections[5].reference}
    }};
    std::array<dependency::ReadDependency, 2> reads_c{{
        {sources[1].reference, selections[3].reference},
        {sources[2].reference, selections[4].reference}
    }};
    std::array<dependency::ReadDependency, 1> reads_zero{{
        {sources[3].reference, selections[7].reference}
    }};
    std::array<dependency::ReadDependency, 1> reads_empty_selection{{
        {sources[0].reference, selections[6].reference}
    }};

    std::array<execution::ConditionDefinition, 6> definitions{};

    Fixture() {
        definitions = {{
            {{{201, 1}, reads_a}, true},
            {{{202, 1}, reads_b}, false},
            {{{203, 1}, reads_c}, true},
            {{{204, 1}, {}}, false},
            {{{205, 1}, reads_zero}, false},
            {{{206, 1}, reads_empty_selection}, false}
        }};
    }

    std::shared_ptr<const execution::PreparedConditionIndex> prepare() const {
        return std::make_shared<execution::PreparedConditionIndex>(
            structure::structure_epoch{7}, sources, selections, definitions);
    }

    std::array<execution::SourceRevision, 4> revisions() const {
        return {{{sources[0].reference, 10}, {sources[1].reference, 20},
                 {sources[2].reference, 30}, {sources[3].reference, 40}}};
    }
};

execution::ConditionState state_for(
    const std::shared_ptr<const execution::PreparedConditionIndex>& index,
    const Fixture& fixture, std::uint64_t time_generation = 4) {
    const auto revisions = fixture.revisions();
    return execution::ConditionState(index, 9, revisions, time_generation);
}

void publish(execution::ConditionState& state, dependency::condition_id id,
             mechanism::ConditionOutcome outcome) {
    const auto evaluation = state.begin(id);
    state.publish(evaluation, outcome);
}

void reverse_index_is_exact_and_metadata_is_owned() {
    Fixture fixture;
    const auto index = fixture.prepare();
    const auto state_ref = fixture.sources[0].reference;
    const auto context_ref = fixture.sources[1].reference;
    const auto parameter_ref = fixture.sources[2].reference;

    const auto left = range(301, fixture.sources[0], 0, 1);
    require_ids(index->affected_conditions(state_ref, left), {201},
                "state subset maps only to the condition reading that coordinate");
    const auto shared_context = range(302, fixture.sources[1], 1, 1);
    require_ids(index->affected_conditions(context_ref, shared_context), {201, 202},
                "shared context source maps to both dependent conditions");
    const auto shared_parameter = range(303, fixture.sources[2], 0, 1);
    require_ids(index->affected_conditions(parameter_ref, shared_parameter), {201, 203},
                "parameter SourceRef is indexed as an ordinary condition input");
    require_ids(index->affected_conditions(fixture.sources[0].reference), {201, 202, 206},
                "whole-source lookup conservatively finds all declared state readers");
    const auto nonempty_state = range(304, fixture.sources[0], 7, 1);
    require_ids(index->affected_conditions(state_ref, nonempty_state), {},
                "an empty declared selection has no overlap with changed coordinates");

    // The explicit selector and read arrays below are mutable caller storage.
    // Preparation must retain their original semantics after that storage changes.
    auto owned_source = source(50, 50, 51, 52, 6);
    std::array<std::uint64_t, 2> coordinates{1, 4};
    auto explicit_selection = all(501, owned_source);
    explicit_selection.kind = dependency::SelectionKind::explicit_coordinates;
    explicit_selection.coordinates = coordinates;
    std::array<dependency::SourceDescriptor, 1> source_catalog{owned_source};
    std::array<dependency::SelectionDescriptor, 1> selection_catalog{explicit_selection};
    std::array<dependency::ReadDependency, 1> explicit_reads{{
        {owned_source.reference, explicit_selection.reference}
    }};
    std::array<execution::ConditionDefinition, 1> condition_catalog{{
        {{{502, 1}, explicit_reads}, false}
    }};
    const execution::PreparedConditionIndex copied(
        {8}, source_catalog, selection_catalog, condition_catalog);
    coordinates = {2, 3};
    explicit_reads[0].source = source(99, 50, 51, 52, 6).reference;
    selection_catalog[0].order = structure::order_id{999, 1};
    source_catalog[0].reference.representation = dependency::representation_id{999, 1};

    auto coordinate_one = all(503, owned_source);
    coordinate_one.kind = dependency::SelectionKind::explicit_coordinates;
    const std::array<std::uint64_t, 1> one{1};
    coordinate_one.coordinates = one;
    auto coordinate_two = coordinate_one;
    coordinate_two.reference.identity = {504, 1};
    const std::array<std::uint64_t, 1> two{2};
    coordinate_two.coordinates = two;
    require_ids(copied.affected_conditions(owned_source.reference, coordinate_one), {502},
                "prepared explicit membership survives caller coordinate mutation");
    require_ids(copied.affected_conditions(owned_source.reference, coordinate_two), {},
                "prepared selector does not adopt later caller membership");
    require(dependency::same_source(copied.sources().front().reference,
                                    owned_source.reference),
            "source catalog is copied with its original representation identity");
}

void invalidation_preserves_unaffected_cache_and_pending_inactive_work() {
    Fixture fixture;
    const auto index = fixture.prepare();
    auto state = state_for(index, fixture);
    const dependency::condition_id a{201, 1};
    const dependency::condition_id b{202, 1};
    const dependency::condition_id c{203, 1};
    const dependency::condition_id empty{204, 1};
    const dependency::condition_id zero{205, 1};
    const dependency::condition_id empty_selection{206, 1};

    publish(state, a, mechanism::ConditionOutcome::excluded);
    publish(state, b, mechanism::ConditionOutcome::eligible);
    publish(state, c, mechanism::ConditionOutcome::unresolved);
    publish(state, empty, mechanism::ConditionOutcome::excluded);
    publish(state, zero, mechanism::ConditionOutcome::eligible);
    publish(state, empty_selection, mechanism::ConditionOutcome::excluded);
    require(!state.dirty(a) && !state.dirty(b) && !state.dirty(c),
            "all three explicit outcomes become cached current results");
    require(state.cached(c) == mechanism::ConditionOutcome::unresolved,
            "unresolved is a distinct publishable result rather than an exclusion");
    const auto cached_a = state.acquire_cached(a);
    const auto cached_b = state.acquire_cached(b);
    require(cached_a && cached_b && state.is_current(cached_a->evaluation)
                && state.is_current(cached_b->evaluation),
            "acquired cached outcomes carry current owner/source stamps");

    // c1 is intentionally inactive while changed work is recorded.
    state.set_active(a, false);
    const auto stale_b = state.begin(b);
    const auto state_left = range(601, fixture.sources[0], 0, 1);
    const auto invalidated = state.source_changed(fixture.sources[0].reference,
                                                   11, state_left);
    require_ids(invalidated, {201}, "coordinate write invalidates only overlapping read");
    require(state.dirty(a) && !state.dirty(b) && !state.dirty(c),
            "non-overlapping state and unrelated context/parameter caches stay current");
    require(!state.is_current(cached_a->evaluation)
                && !state.is_current(cached_b->evaluation),
            "held tickets cannot silently survive source generation changes");
    const auto refreshed_b = state.acquire_cached(b);
    require(refreshed_b && state.is_current(refreshed_b->evaluation)
                && refreshed_b->outcome == mechanism::ConditionOutcome::eligible,
            "unaffected cached outcome can rebind to current source generation without reevaluation");
    require(!state.acquire_cached(a), "dirty conditions cannot issue a cached evaluation");
    require(!state.active(a)
                && ids(state.dirty_conditions(true)).empty()
                && ids(state.dirty_conditions(false)) == std::vector<std::uint64_t>{201},
            "inactive invalidated condition remains pending until reactivated");
    require(throws<std::logic_error>([&] {
                state.publish(stale_b, mechanism::ConditionOutcome::excluded);
            }),
            "a generation change conservatively rejects an in-flight reader of that source");
    require(state.cached(b) == mechanism::ConditionOutcome::eligible && !state.dirty(b),
            "rejected stale publication cannot overwrite an unaffected cached result");
    state.set_active(a, true);
    require(ids(state.dirty_conditions(true)) == std::vector<std::uint64_t>{201},
            "reactivating an invalidated condition exposes its pending evaluation");
    publish(state, a, mechanism::ConditionOutcome::eligible);

    const auto context_shared = range(602, fixture.sources[1], 1, 1);
    require_ids(state.source_changed(fixture.sources[1].reference, 21, context_shared),
                {201, 202}, "shared condition-only context change invalidates both readers");
    require(state.dirty(a) && state.dirty(b) && !state.dirty(c),
            "condition-only sources invalidate exactly their declared readers");
    publish(state, a, mechanism::ConditionOutcome::eligible);
    publish(state, b, mechanism::ConditionOutcome::excluded);

    require_ids(state.time_changed(5), {201, 203},
                "temporal revision invalidates only time-sensitive conditions");
    require(state.dirty(a) && !state.dirty(b) && state.dirty(c),
            "time invalidation is independent of numeric source writes");
    publish(state, a, mechanism::ConditionOutcome::excluded);
    publish(state, c, mechanism::ConditionOutcome::eligible);

    const auto parameter_low = range(603, fixture.sources[2], 0, 1);
    require_ids(state.source_changed(fixture.sources[2].reference, 31, parameter_low),
                {201, 203}, "shared parameter source invalidates conditions that read its subset");
    require(state.dirty(a) && !state.dirty(b) && state.dirty(c),
            "parameter subset leaves the disjoint parameter reader cached");

    require(throws<std::logic_error>([&] {
                (void)state.source_changed(fixture.sources[2].reference, 31, parameter_low);
            }),
            "equal source generation is rejected");
    require(throws<std::logic_error>([&] {
                (void)state.source_changed(fixture.sources[2].reference, 30, parameter_low);
            }),
            "backward source generation is rejected");
    require(state.source_generation(fixture.sources[2].reference) == 31,
            "rejected generation updates leave the current revision intact");

    // Whole-source invalidation is conservative, including a valid empty axis;
    // a condition with no declared reads remains independent of all sources.
    require_ids(state.source_changed(fixture.sources[3].reference, 41), {205},
                "zero-sized source catalog is valid and supports whole-source invalidation");
    require(!state.dirty(empty), "empty-read condition is unaffected by unrelated writes");
    require_ids(state.source_changed(fixture.sources[0].reference, 12), {201, 202, 206},
                "whole-source change conservatively includes a declared empty selection");
}

void tickets_and_associations_fail_closed() {
    Fixture fixture;
    const auto index = fixture.prepare();
    auto state = state_for(index, fixture);
    const auto other_state_revisions = fixture.revisions();
    execution::ConditionState other(index, 9, other_state_revisions, 4);
    const dependency::condition_id a{201, 1};

    const auto superseded = state.begin(a);
    const auto current = state.begin(a);
    require(throws<std::logic_error>([&] {
                state.publish(superseded, mechanism::ConditionOutcome::eligible);
            }),
            "a newer evaluation ticket supersedes an older ticket");
    require(!state.cached(a) && state.dirty(a),
            "superseded publication cannot mark a condition current");
    state.publish(current, mechanism::ConditionOutcome::excluded);
    require(state.cached(a) == mechanism::ConditionOutcome::excluded,
            "the current ticket publishes the explicit outcome");
    require(throws<std::logic_error>([&] {
                state.publish(current, mechanism::ConditionOutcome::eligible);
            }),
            "a published ticket cannot publish a second, conflicting outcome");
    require(state.cached(a) == mechanism::ConditionOutcome::excluded,
            "duplicate publication cannot overwrite the already-published result");

    const auto foreign = other.begin(a);
    require(throws<std::logic_error>([&] {
                state.publish(foreign, mechanism::ConditionOutcome::eligible);
            }),
            "a ticket from another state owner is rejected");
    require(state.cached(a) == mechanism::ConditionOutcome::excluded,
            "foreign publication does not alter the existing result");

    const auto time_stale = state.begin(a);
    require_ids(state.time_changed(5), {201, 203}, "time update publishes affected IDs");
    require(throws<std::logic_error>([&] {
                state.publish(time_stale, mechanism::ConditionOutcome::eligible);
            }),
            "temporal revision makes an in-flight time-sensitive ticket stale");
    require(!state.cached(a) && state.dirty(a),
            "stale time result cannot repopulate an invalidated cache");

    auto wrong_order = range(701, fixture.sources[0], 0, 1);
    wrong_order.order = structure::order_id{999, 1};
    require(throws<std::invalid_argument>([&] {
                (void)index->affected_conditions(fixture.sources[0].reference,
                                                 wrong_order);
            }),
            "same-size changed selector in another logical order is rejected");
    auto wrong_domain = range(702, fixture.sources[0], 0, 1);
    wrong_domain.reference.domain = structure::domain_id{999, 1};
    require(throws<std::invalid_argument>([&] {
                (void)index->affected_conditions(fixture.sources[0].reference,
                                                 wrong_domain);
            }),
            "same-size changed selector in another domain is rejected");
    auto wrong_representation = fixture.sources[0].reference;
    wrong_representation.representation = dependency::representation_id{999, 1};
    require(throws<std::invalid_argument>([&] {
                (void)index->affected_conditions(wrong_representation);
            }),
            "same source identity with another representation is rejected");

    const auto generation_before_bad_selector = state.source_generation(
        fixture.sources[0].reference);
    require(throws<std::invalid_argument>([&] {
                (void)state.source_changed(fixture.sources[0].reference,
                                           generation_before_bad_selector + 1,
                                           wrong_order);
            }),
            "malformed changed selection is rejected before publishing its generation");
    require(state.source_generation(fixture.sources[0].reference)
                == generation_before_bad_selector,
            "invalid changed selection does not mutate the source revision");
}

void state_retains_prepared_index_storage() {
    Fixture fixture;
    const auto revisions = fixture.revisions();
    auto index_owner = std::make_shared<execution::PreparedConditionIndex>(
        structure::structure_epoch{88}, fixture.sources, fixture.selections,
        fixture.definitions);
    execution::ConditionState state(index_owner, 13, revisions, 6);
    const auto evaluation = state.begin({201, 1});

    execution::PreparedConditionIndex moved(std::move(*index_owner));
    require(index_owner->condition_count() == 0 && index_owner->sources().empty()
                && index_owner->structure_epoch().value == 0,
            "moved-from index exposes an empty, invalid snapshot");
    require(throws<std::logic_error>([&] {
                (void)index_owner->affected_conditions(fixture.sources[0].reference);
            }),
            "moved-from index rejects queries instead of dereferencing empty storage");
    require(throws<std::invalid_argument>([&] {
                execution::ConditionState invalid(index_owner, 13, revisions, 6);
            }),
            "a moved-from prepared index cannot seed a new execution state");
    index_owner.reset();

    // The execution state retained the immutable index contents independently
    // of the PreparedConditionIndex object's lifetime and move state.
    state.publish(evaluation, mechanism::ConditionOutcome::eligible);
    require(state.cached({201, 1}) == mechanism::ConditionOutcome::eligible
                && state.structure_epoch().value == 88
                && moved.structure_epoch().value == 88,
            "existing state remains usable after its prepared index object moves and expires");
}

void invalid_catalogs_and_empty_axes_are_checked() {
    Fixture fixture;
    const auto index = fixture.prepare();
    const auto revisions = fixture.revisions();

    require(throws<std::invalid_argument>([&] {
                execution::ConditionState invalid(index, 0, revisions, 4);
            }),
            "binding revision zero is malformed");
    require(throws<std::invalid_argument>([&] {
                std::array<execution::SourceRevision, 3> incomplete{
                    revisions[0], revisions[1], revisions[2]};
                execution::ConditionState invalid(index, 9, incomplete, 4);
            }),
            "revision catalog must cover every prepared source exactly once");
    require(throws<std::invalid_argument>([&] {
                auto duplicate = revisions;
                duplicate[1] = duplicate[0];
                execution::ConditionState invalid(index, 9, duplicate, 4);
            }),
            "duplicate source revision is malformed");
    require(throws<std::invalid_argument>([&] {
                auto wrong_domain = revisions;
                wrong_domain[0].source.domain = structure::domain_id{999, 1};
                execution::ConditionState invalid(index, 9, wrong_domain, 4);
            }),
            "same source identity with wrong domain cannot bind a generation");
    require(throws<std::invalid_argument>([&] {
                auto wrong_representation = revisions;
                wrong_representation[0].source.representation =
                    dependency::representation_id{999, 1};
                execution::ConditionState invalid(index, 9, wrong_representation, 4);
            }),
            "same source identity with wrong representation cannot bind a generation");
    require(throws<std::invalid_argument>([&] {
                (void)std::make_shared<execution::PreparedConditionIndex>(
                    structure::structure_epoch{0}, fixture.sources,
                    fixture.selections, fixture.definitions);
            }),
            "invalid structure epoch cannot prepare a condition index");

    auto bad_selections = fixture.selections;
    bad_selections[0].order = structure::order_id{999, 1};
    require(throws<std::invalid_argument>([&] {
                execution::PreparedConditionIndex invalid(
                    structure::structure_epoch{9}, fixture.sources,
                    bad_selections, fixture.definitions);
            }),
            "a same-size selector in the wrong logical order cannot enter a prepared index");

    Fixture empty_fixture;
    const auto empty_index = empty_fixture.prepare();
    auto empty_state = state_for(empty_index, empty_fixture);
    publish(empty_state, {204, 1}, mechanism::ConditionOutcome::excluded);
    publish(empty_state, {205, 1}, mechanism::ConditionOutcome::eligible);
    require(!empty_state.dirty({204, 1}) && !empty_state.dirty({205, 1}),
            "empty read sets and all-selection zero axes can be evaluated and cached");
}

void condition_index_tracks_the_prepared_gated_product_binding() {
    std::array<mechanism::GatedProductGroup, 2> groups{{
        {0, 2, 0, 2, 0, 0, 10},
        {2, 3, 2, 2, 1, 1, 20}
    }};
    std::array<std::uint64_t, 5> arguments{{0, 2, 2, 4, 0}};
    std::array<mechanism::OutputIncidence, 4> outputs{{
        {0, 1.0f}, {2, 0.5f}, {4, 1.0f}, {0, 0.25f}
    }};
    const mechanism::GatedProductTopologyView topology{
        groups.data(), groups.size(), arguments.data(), arguments.size(),
        outputs.data(), outputs.size(), 5, 2, 2};
    const auto make_source = [](std::uint64_t id, std::uint64_t domain,
                                std::uint64_t order, std::uint64_t rep,
                                std::uint64_t extent) {
        const structure::domain_id d{domain, 1};
        return dependency::SourceDescriptor{
            {dependency::source_id{id, 1}, d,
             dependency::representation_id{rep, 1}},
            {d, structure::order_id{order, 1}, extent}};
    };
    const mechanism::GatedProductSourceBindings sources{
        make_source(301, 11, 101, 201, 5),
        make_source(302, 12, 102, 202, 4),
        make_source(303, 13, 103, 203, 2),
        make_source(304, 13, 103, 204, 2),
        make_source(305, 13, 103, 205, 2),
        make_source(306, 11, 101, 206, 5)
    };
    const dependency::Applicability applicability{
        {401, 1}, {402, 1}, {403, 1}, {404, 1}, {1}};
    const mechanism::PreparedGatedProductBinding binding(
        topology, sources, applicability, {405, 1}, {406, 1}, {407, 1});

    const auto bound_conditions = binding.foundation().condition_reads;
    require(bound_conditions.size() == 2,
            "prepared operation supplies one read declaration per gate");
    const std::array<execution::ConditionDefinition, 2> conditions{{
        {bound_conditions[0], true},
        {bound_conditions[1], false}
    }};
    auto index = std::make_shared<execution::PreparedConditionIndex>(
        structure::structure_epoch{77}, binding.sources(), binding.selections(),
        conditions);
    std::array<execution::SourceRevision, 6> revisions{};
    for (std::size_t i = 0; i < binding.sources().size(); ++i)
        revisions[i] = {binding.sources()[i].reference, 100 + i};
    execution::ConditionState state(index, 12, revisions, 3);

    const auto condition0 = binding.condition(0);
    const auto condition1 = binding.condition(1);
    publish(state, condition0, mechanism::ConditionOutcome::excluded);
    publish(state, condition1, mechanism::ConditionOutcome::eligible);
    const auto current_gate = state.acquire_cached(condition0);
    require(current_gate && state.is_current(current_gate->evaluation)
                && binding.allows_inactive(0, mechanism::Omission::input_read,
                                           current_gate->outcome),
            "published fresh gate outcome joins the operation's exact omission rule");

    const auto& regulator_source = binding.source(mechanism::GatedProductSource::regulators);
    auto regulator_zero = range(800, regulator_source, 0, 2);
    require_ids(state.source_changed(regulator_source.reference, 200, regulator_zero),
                {condition0.low},
                "prepared regulator selection invalidates only its gate's condition");
    require(state.dirty(condition0) && !state.dirty(condition1)
                && state.cached(condition1) == mechanism::ConditionOutcome::eligible,
            "other gate cache survives a non-overlapping regulator update");
    require(!state.is_current(current_gate->evaluation)
                && !state.acquire_cached(condition0),
            "a held omission outcome is stale after its declared input changes");

    const auto temporal = state.time_changed(4);
    require_ids(temporal, {condition0.low},
                "temporal-only gate dependency invalidates without a source write");
    require(state.dirty(condition0) && !state.dirty(condition1),
            "runtime freshness stays associated with each prepared gate condition");
}

} // namespace

int main() {
    try {
        reverse_index_is_exact_and_metadata_is_owned();
        invalidation_preserves_unaffected_cache_and_pending_inactive_work();
        tickets_and_associations_fail_closed();
        invalid_catalogs_and_empty_axes_are_checked();
        state_retains_prepared_index_storage();
        condition_index_tracks_the_prepared_gated_product_binding();
        std::cout << "condition index tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "condition index test failed: " << error.what() << '\n';
        return 1;
    }
}
