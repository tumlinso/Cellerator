#include <Cellerator/execution/dependency/packed_program.cuh>
#include <Cellerator/execution/atom_fragment/external_plane_binding_v1.hh>
#include <Cellerator/compute/operation/candidate_catalog_v3/catalog.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace cellerator::execution::dependency {
namespace bm = tensor::block_masked;
namespace op = ops::mechanism;
namespace jc = preparation;
namespace af = atom_fragment;
namespace cat = compute::operation::catalog_v3;
namespace pg = program;
namespace {
// Registry identifiers for this specific versioned format/operation, not
// identifiers of a biological source or of any individual allocation.
constexpr std::uint64_t provider = 0x43454c4c45524154ull;
constexpr jc::persistent_identity_v1 species{provider, 0x424d465031ull};
constexpr jc::persistent_identity_v1 plane{provider, 0x4650313656414cull};
constexpr jc::persistent_identity_v1 encoding{provider, 0x424d465031ull};
constexpr jc::persistent_identity_v1 projection{provider, 0x4750504d415031ull};
constexpr std::uint64_t candidate_id = 0x47505041434b31ull;
constexpr std::uint64_t stage_id = 0x47505053544131ull;

bool same_layout(const bm::LayoutView<>& a, const bm::LayoutView<>& b) noexcept {
    return a.tile_block_offsets == b.tile_block_offsets && a.block_ids == b.block_ids
        && a.participating_row_masks == b.participating_row_masks
        && a.block_row_entry_offsets == b.block_row_entry_offsets
        && a.feature_masks == b.feature_masks && a.row_block_value_offsets == b.row_block_value_offsets
        && a.feature_block_offsets == b.feature_block_offsets && a.tile_row_width == b.tile_row_width
        && a.rows == b.rows && a.features == b.features && a.tiles == b.tiles
        && a.stored_values == b.stored_values && a.feature_blocks == b.feature_blocks
        && a.tile_blocks == b.tile_blocks && a.row_block_entries == b.row_block_entries;
}
bool same_prepared_input(const op::PackedGatedProductInput<std::uint16_t>& a,
                         const op::PackedGatedProductInput<std::uint16_t>& b) noexcept {
    const auto& x = a.topology; const auto& y = b.topology;
    return same_layout(a.x.layout, b.x.layout) && x.groups == y.groups && x.group_count == y.group_count
        && x.arguments == y.arguments && x.argument_count == y.argument_count && x.outputs == y.outputs
        && x.output_count == y.output_count && x.coordinates == y.coordinates && x.conditions == y.conditions
        && x.parameters == y.parameters && a.argument_value_offsets == b.argument_value_offsets
        && a.coordinate_value_offsets == b.coordinate_value_offsets && a.batch == b.batch
        && a.include_baseline == b.include_baseline && a.beta == b.beta;
}
void require(bool value, const char* message) { if (!value) throw std::invalid_argument(message); }
} // namespace

struct PreparedPackedProgram::Impl {
    struct Stage { const Impl* owner; bool backward; };
    std::shared_ptr<const op::PreparedGatedProductBinding> semantics;
    std::shared_ptr<const bm::DeviceStructure> structure;
    PackedAtomIdentities ids;
    structure::structure_id structure_id{};
    structure::order_id physical_order{};
    std::uint64_t epoch = 0, placement_revision = 0;
    op::PackedGatedProductInput<std::uint16_t> prototype{};
    std::vector<std::uint64_t> coverage;
    jc::atom_requirement_v1 requirement{};
    af::atom_bound_candidate_v1 candidate{};
    Stage stage_states[2]{{this, false}, {this, true}};
    pg::prepared_stage_v2 stages[2]{};
    pg::prepared_program_v2 programs[2]{};
    af::prepared_atom_fragment_v1 fragments[2]{};

    explicit Impl(const PackedProgramPreparation& p)
        : semantics(p.semantics), structure(p.structure), ids(p.identities),
          structure_id(p.structure_identity), physical_order(p.physical_order),
          epoch(p.structure_epoch), placement_revision(p.placement_revision), prototype(p.prepared_input) {
        require(semantics && structure && structure::valid_identity(structure_id)
            && structure::valid_identity(physical_order) && epoch && placement_revision,
            "invalid packed program preparation");
        for (auto id : {ids.atom, ids.requirement, ids.coverage, ids.affordance, ids.binding})
            require(bool(jc::validate_persistent_identity_v1(id)), "invalid packed atom identity");
        const auto full = semantics->topology();
        require(!p.group_indices.empty() && prototype.topology.group_count == p.group_indices.size()
            && prototype.batch == structure->host().rows() && prototype.topology.coordinates == full.coordinates
            && prototype.topology.conditions == full.conditions && prototype.topology.parameters == full.parameters
            && prototype.topology.argument_count == full.argument_count && prototype.topology.output_count == full.output_count
            && same_layout(prototype.x.layout, structure->view()), "packed program shape mismatch");
        require(semantics->coordinate_binding().valid() && semantics->source(op::GatedProductSource::state).axis.extent == full.coordinates,
            "invalid packed semantic axis");
        std::vector<op::GatedProductGroup> groups; groups.reserve(p.group_indices.size());
        const auto has_state_read = [&](op::Phase phase,
                                        const geometry::dependency::SelectionDescriptor& selection) {
            const auto& state = semantics->source(op::GatedProductSource::state);
            const auto reads = semantics->phase(phase).reads;
            return std::any_of(reads.begin(), reads.end(), [&](const auto& read) {
                return geometry::dependency::same_source(read.source, state.reference)
                    && geometry::dependency::same_selection(read.selection, selection.reference);
            });
        };
        for (auto i : p.group_indices) {
            require(i < full.group_count, "group index outside semantic topology");
            require(std::count(p.group_indices.begin(), p.group_indices.end(), i) == 1,
                "duplicate packed program group index");
            const auto g = full.groups[i]; groups.push_back(g);
            require(g.output_count != 0, "packed program cannot execute an outputless group");
            const auto& selection = semantics->group_participant_selection(i);
            for (auto phase : {op::Phase::primal, op::Phase::input_vjp,
                               op::Phase::condition_vjp, op::Phase::parameter_vjp})
                require(has_state_read(phase, selection), "group state read absent from semantic phase");
            for (std::uint64_t j = g.argument_begin; j < g.argument_begin + g.argument_count; ++j)
            {
                require(geometry::dependency::selection_contains(selection, full.arguments[j]),
                    "group argument absent from semantic selection");
                coverage.push_back(full.arguments[j]);
            }
        }
        auto local = full; local.groups = groups.data(); local.group_count = groups.size();
        // Check every physical participant slot for every row against the exact
        // semantic group selection; missing bytes are never implicit zero.
        (void)op::prepare_argument_value_offsets(structure->host(), local);
        if (prototype.include_baseline) {
            require(has_state_read(op::Phase::primal,
                semantics->selection(op::GatedProductSource::state)),
                "baseline state read absent from semantic phase");
            (void)op::prepare_coordinate_value_offsets(structure->host(), full.coordinates);
            coverage.resize(full.coordinates); std::iota(coverage.begin(), coverage.end(), std::uint64_t{0});
        } else {
            std::sort(coverage.begin(), coverage.end());
            coverage.erase(std::unique(coverage.begin(), coverage.end()), coverage.end());
        }
        // v1's physical extent schema cannot bind an empty allocation.
        require(!coverage.empty() && structure->view().stored_values, "empty packed atom coverage");
        requirement.requirement_identity = ids.requirement; requirement.exact_coverage_identity = ids.coverage;
        requirement.accepted_atom_species = &species; requirement.accepted_atom_species_count = 1;
        requirement.required_planes = &plane; requirement.required_plane_count = 1;
        requirement.numeric = {numeric_type::f16, numeric_type::f32, numeric_type::f32, 0};
        requirement.index_width = layout::LocalIndexWidth::u32;
        requirement.required_order = physical_order; requirement.minimum_alignment = alignof(std::uint16_t);
        requirement.contiguity = jc::contiguity_requirement_v1::contiguous;
        requirement.mutability = jc::mutability_requirement_v1::mutable_value_generation;
        require(bool(jc::validate_atom_requirement_v1(requirement)), "invalid packed atom requirement");
        cat::candidate_stage_v3 kernel{}; kernel.stage_id = stage_id; kernel.kernel_id = stage_id;
        std::strcpy(kernel.stable_name, "packed-gated-product-fp16-fp32");
        cat::candidate_descriptor_v3 descriptor{};
        descriptor.identity.candidate_id = candidate_id; descriptor.identity.provider_id = provider;
        descriptor.identity.operation_id = candidate_id; descriptor.identity.width_min = 1;
        descriptor.identity.width_max = 32; descriptor.stages = &kernel; descriptor.stage_count = 1;
        const cat::candidate_catalog_view_v3 catalog{&descriptor, 1};
        const jc::atom_binding_request_v1 binding{ids.atom, ids.requirement, ids.affordance};
        std::uint64_t written = 0;
        require(bool(af::discover_atom_bound_candidates_v1(catalog, &requirement, 1, &binding, 1,
            &candidate, 1, &written)) && written == 1, "failed packed candidate preparation");
        for (unsigned i = 0; i < 2; ++i) {
            stages[i].stable_stage_id = stage_id; stages[i].candidate_id = candidate_id;
            stages[i].prepared_state = &stage_states[i]; stages[i].launch = launch;
            stages[i].admit = admit;
            programs[i].stages = &stages[i]; programs[i].stage_count = 1;
            require(bool(af::prepare_atom_fragment_v1(candidate, programs[i], physical_order,
                semantics->source(op::GatedProductSource::result).axis.logical_order, &fragments[i])),
                "failed packed fragment preparation");
        }
    }

    pg::program_status admission(const PackedProgramLaunch& l, bool backward, void* stream_handle) const noexcept {
        try {
            const auto& declared = semantics->source(op::GatedProductSource::state);
            if (!l.source || !geometry::dependency::same_source(l.source->reference, declared.reference)
                || !structure::same_identity(l.source->axis.domain, declared.axis.domain)
                || !structure::same_identity(l.source->axis.logical_order, declared.axis.logical_order)
                || l.source->axis.extent != declared.axis.extent
                || !l.read || !l.readiness || !l.stream || !l.readiness->recorded()
                || l.readiness->device() != structure->device() || l.stream->device() != structure->device())
                return pg::program_status::launch_failed;
            // view() rejects a moved-from lease; device()/structure() assume a
            // live handle, so check the view before accessing those members.
            const auto retained = l.read->view();
            if (reinterpret_cast<void*>(l.stream->get()) != stream_handle
                || l.read->device() != structure->device() || &l.read->structure() != structure.get()
                || l.read->generation() != l.expected.value_generation
                || !same_prepared_input(l.input, prototype) || l.input.x.values != retained.values
                || !same_layout(retained.layout, prototype.x.layout)
                || l.state.data != l.input.x.values || l.capacities.values > l.state.bytes
                || l.capacities.values < structure->view().stored_values * sizeof(std::uint16_t)
                || l.state.bytes < structure->view().stored_values * sizeof(std::uint16_t)
                || !validate_extent(l.expected, l.state)
                || !structure::same_identity(l.expected.domain, semantics->source(op::GatedProductSource::state).axis.domain)
                || !structure::same_identity(l.expected.logical_order, semantics->source(op::GatedProductSource::state).axis.logical_order)
                || !structure::same_identity(l.expected.physical_order, physical_order)
                || !structure::same_identity(l.expected.structure, structure_id)
                || l.expected.structure_epoch != epoch || l.expected.placement_revision != placement_revision
                || l.expected.device != structure->device() || !l.status
                || !l.saved.gate_arguments || !l.saved.products
                || (backward ? !l.output_gradient : !l.output)) return pg::program_status::launch_failed;
            const bool valid_buffers = backward
                ? op::validate_packed_backward_buffers(l.input, l.capacities, l.output_gradient,
                    l.gradients, l.saved, l.status)
                : op::validate_packed_forward_buffers(l.input, l.capacities, l.output, l.saved, l.status);
            if (!valid_buffers) return pg::program_status::launch_failed;
            const auto generation = l.expected.value_generation;
            jc::atom_plane_affordance_v1 plane_description{};
            plane_description.plane_identity = plane; plane_description.order = physical_order;
            plane_description.storage = numeric_type::f16; plane_description.logical = numeric_type::f32;
            plane_description.mutability = jc::mutability_requirement_v1::mutable_value_generation;
            plane_description.generation = {generation};
            jc::atom_affordance_v1 affordance{};
            affordance.affordance_identity = ids.affordance; affordance.atom_species = species;
            affordance.exact_coverage_identity = ids.coverage; affordance.physical_encoding = encoding;
            affordance.local_projection_abi = projection; affordance.planes = &plane_description; affordance.plane_count = 1;
            jc::external_extent_v1 extent{};
            extent.address = l.state.data; extent.location = {residency_kind::device, {}, structure->device(), 1};
            extent.bytes = l.state.bytes; extent.alignment = alignof(std::uint16_t); extent.order = physical_order;
            extent.generation = {generation};
            extent.readiness = {reinterpret_cast<std::uintptr_t>(l.readiness->get()), generation};
            extent.lease = {reinterpret_cast<std::uintptr_t>(l.read), generation};
            jc::external_binding_v1 binding{};
            binding.binding_identity = ids.binding; binding.atom_identity = ids.atom; binding.plane_identity = plane;
            binding.extents = &extent; binding.extent_count = 1; binding.total_bytes = extent.bytes;
            af::bound_atom_extent_v1 bound{}; std::uint64_t written = 0;
            if (!af::bind_external_atom_planes_v1(fragments[backward ? 1 : 0], requirement, affordance,
                    &binding, 1, &bound, 1, &written) || written != 1 || bound.address != l.input.x.values)
                return pg::program_status::launch_failed;
            return pg::program_status::success;
        } catch (...) { return pg::program_status::launch_failed; }
    }
    static pg::program_status admit(const void* state, const pg::launch_binding_v2& b, void* stream) noexcept {
        if (!state || !b.values) return pg::program_status::invalid_argument;
        const auto& stage = *static_cast<const Stage*>(state);
        return stage.owner->admission(*static_cast<const PackedProgramLaunch*>(b.values), stage.backward, stream);
    }
    static pg::program_status launch(const void* state, const pg::launch_binding_v2& b, void*) noexcept {
        try {
            const auto& stage = *static_cast<const Stage*>(state);
            const auto& l = *static_cast<const PackedProgramLaunch*>(b.values);
            // The batch may visit several devices on one host thread. Select
            // this stream's device for both the wait and kernel launch.
            memory::cuda::DeviceGuard guard(l.stream->device());
            l.stream->wait(*l.readiness);
            auto status = stage.backward
                ? op::launch_packed_backward_checked(l.input, l.capacities, l.output_gradient, l.gradients, l.saved, l.status, l.stream->get())
                : op::launch_packed_forward_checked(l.input, l.capacities, l.output, l.saved, l.status, l.stream->get());
            return status == cudaSuccess ? pg::program_status::success : pg::program_status::launch_failed;
        } catch (...) { return pg::program_status::launch_failed; }
    }
    pg::program_status execute(const PackedProgramLaunch& l, bool backward) const noexcept {
        if (!l.stream) return pg::program_status::invalid_argument;
        pg::launch_binding_v2 binding{}; binding.values = &l;
        return pg::execute_prepared_program_v2(programs[backward ? 1 : 0], &binding, 1,
            reinterpret_cast<void*>(l.stream->get()));
    }
};

PreparedPackedProgram::PreparedPackedProgram(const PackedProgramPreparation& p) : impl_(std::make_unique<Impl>(p)) {}
PreparedPackedProgram::~PreparedPackedProgram() = default;
pg::program_status PreparedPackedProgram::admit_forward(const PackedProgramLaunch& l) const noexcept {
    return impl_->admission(l, false, l.stream ? reinterpret_cast<void*>(l.stream->get()) : nullptr);
}
pg::program_status PreparedPackedProgram::admit_backward(const PackedProgramLaunch& l) const noexcept {
    return impl_->admission(l, true, l.stream ? reinterpret_cast<void*>(l.stream->get()) : nullptr);
}
pg::program_status PreparedPackedProgram::forward(const PackedProgramLaunch& l) const noexcept { return impl_->execute(l, false); }
pg::program_status PreparedPackedProgram::backward(const PackedProgramLaunch& l) const noexcept { return impl_->execute(l, true); }
const jc::atom_requirement_v1& PreparedPackedProgram::requirement() const noexcept { return impl_->requirement; }
const af::prepared_atom_fragment_v1& PreparedPackedProgram::forward_fragment() const noexcept { return impl_->fragments[0]; }
const af::prepared_atom_fragment_v1& PreparedPackedProgram::backward_fragment() const noexcept { return impl_->fragments[1]; }
std::span<const std::uint64_t> PreparedPackedProgram::coverage_coordinates() const noexcept { return impl_->coverage; }

pg::program_status admit_packed_programs(std::span<const PackedProgramInvocation> invocations,
                                         bool backward) noexcept {
    for (const auto& invocation : invocations) {
        if (!invocation.program || !invocation.launch) return pg::program_status::invalid_argument;
        const auto status = backward ? invocation.program->admit_backward(*invocation.launch)
                                     : invocation.program->admit_forward(*invocation.launch);
        if (status != pg::program_status::success) return status;
    }
    return pg::program_status::success;
}

pg::program_status execute_packed_programs(std::span<const PackedProgramInvocation> invocations,
                                           bool backward) noexcept {
    const auto admission = admit_packed_programs(invocations, backward);
    if (admission != pg::program_status::success) return admission;
    for (const auto& invocation : invocations) {
        const auto status = backward ? invocation.program->backward(*invocation.launch)
                                     : invocation.program->forward(*invocation.launch);
        if (status != pg::program_status::success) return status;
    }
    return pg::program_status::success;
}
} // namespace cellerator::execution::dependency
