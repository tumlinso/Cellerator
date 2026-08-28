#include "preprocess_internal.cuh"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>

namespace cellerator::compute::preprocess {

void bind_fleet_result(preprocess_fleet_workspace *fleet, unsigned int leader_index, unsigned int cols, preprocess_fleet_result *out) {
    if (out == nullptr) return;
    out->slot_count = fleet != nullptr ? fleet->slot_count : 0u;
    out->leader_index = leader_index;
    out->slot_results = fleet != nullptr ? fleet->results : nullptr;
    if (fleet != nullptr && fleet->devices != nullptr && leader_index < fleet->slot_count) {
        bind_gene_metrics(fleet->devices + leader_index, cols, &out->reduced_gene);
    } else {
        out->reduced_gene = gene_metrics_view{};
    }
}

namespace {

int selected_device_id(const preprocess_fleet_workspace *fleet, unsigned int index) {
    if (fleet == nullptr || index >= fleet->slot_count || fleet->slots == nullptr) return -1;
    return cs_compute_runtime::fleet_device_id(fleet->fleet, fleet->slots[index]);
}

int reduce_sum_to_leader_f32(preprocess_fleet_workspace *fleet,
                             float *const *partials,
                             std::size_t count,
                             unsigned int leader_index,
                             float *leader_out) {
    if (fleet == nullptr || partials == nullptr || leader_index >= fleet->slot_count || (leader_out == nullptr && count != 0u)) return 0;
    if (fleet->slot_count == 0u || count == 0u) return 1;
    cs_compute_runtime::reduce_sum_to_leader_f32_options options{};
#if CELLERATOR_DIST_HAS_NCCL
    options.ranked_nccl = fleet->ranked_nccl.ready != 0u ? &fleet->ranked_nccl : nullptr;
#endif
    try {
        cs_compute_runtime::reduce_sum_to_leader_f32(&fleet->fleet,
                                                     fleet->slots,
                                                     fleet->slot_count,
                                                     (const float *const *) partials,
                                                     count,
                                                     fleet->slots[leader_index],
                                                     leader_out,
                                                     &options);
    } catch (const std::exception &exc) {
        std::fprintf(stderr, "Cellerator preprocess reduction error: %s\n", exc.what());
        return 0;
    }
    return 1;
}

} // namespace

int reduce_gene_metrics_to_leader(preprocess_fleet_workspace *fleet, unsigned int cols, unsigned int leader_index) {
    if (fleet == nullptr || cols == 0u) return fleet != nullptr;
    std::unique_ptr<float *[]> partials(new (std::nothrow) float *[fleet->slot_count]);
    if (!partials) return 0;
    int contiguous = 1;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        preprocess_workspace *workspace = fleet->devices + i;
        const gene_metric_packet_view packet{cols, workspace->gene_sum, workspace->gene_sq_sum, workspace->gene_detected, workspace->active_rows};
        contiguous = contiguous && gene_metric_packet_is_contiguous(&packet, 1u);
    }
    if (contiguous != 0) {
        for (unsigned int i = 0u; i < fleet->slot_count; ++i) partials[i] = fleet->devices[i].gene_sum;
        return reduce_sum_to_leader_f32(fleet,
                                        partials.get(),
                                        gene_metric_packet_float_count(cols, 1u),
                                        leader_index,
                                        fleet->devices[leader_index].gene_sum);
    }
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) partials[i] = fleet->devices[i].gene_sum;
    if (!reduce_sum_to_leader_f32(fleet, partials.get(), cols, leader_index, fleet->devices[leader_index].gene_sum)) return 0;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) partials[i] = fleet->devices[i].gene_sq_sum;
    if (!reduce_sum_to_leader_f32(fleet, partials.get(), cols, leader_index, fleet->devices[leader_index].gene_sq_sum)) return 0;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) partials[i] = fleet->devices[i].gene_detected;
    if (!reduce_sum_to_leader_f32(fleet, partials.get(), cols, leader_index, fleet->devices[leader_index].gene_detected)) return 0;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) partials[i] = fleet->devices[i].active_rows;
    if (!reduce_sum_to_leader_f32(fleet, partials.get(), 1u, leader_index, fleet->devices[leader_index].active_rows)) return 0;
    return 1;
}

void init(preprocess_fleet_workspace *fleet) {
    if (fleet == nullptr) return;
    std::memset(fleet, 0, sizeof(*fleet));
    cs_compute_runtime::init(&fleet->fleet);
#if CELLERATOR_DIST_HAS_NCCL
    cs_dist::init(&fleet->ranked_nccl);
#endif
}

void clear(preprocess_fleet_workspace *fleet) {
    if (fleet == nullptr) return;
    if (fleet->devices != nullptr) {
        for (unsigned int i = 0u; i < fleet->slot_count; ++i) clear(fleet->devices + i);
    }
#if CELLERATOR_DIST_HAS_NCCL
    cs_dist::clear(&fleet->ranked_nccl);
#endif
    std::free(fleet->results);
    std::free(fleet->devices);
    std::free(fleet->slots);
    cs_compute_runtime::clear(&fleet->fleet);
    init(fleet);
}

int setup_fleet(preprocess_fleet_workspace *fleet, const preprocess_fleet_config *config) {
    if (fleet == nullptr) return 0;
    clear(fleet);
    init(fleet);

    const unsigned int stream_flags = config != nullptr ? config->stream_flags : cudaStreamNonBlocking;
    const unsigned int enable_peer = config == nullptr || config->enable_peer_access != 0u;
    if (config != nullptr && config->device_count != 0u && config->device_ids == nullptr) return 0;
    try {
        cs_compute_runtime::discover_fleet(&fleet->fleet, true, stream_flags, enable_peer != 0u);
    } catch (const std::exception &exc) {
        std::fprintf(stderr, "Cellerator preprocess fleet setup error: %s\n", exc.what());
        return 0;
    }
    if (fleet->fleet.local.device_count == 0u) return 0;

    const unsigned int requested = config != nullptr ? config->device_count : 0u;
    const unsigned int selected_count = requested != 0u ? requested : fleet->fleet.local.device_count;
    fleet->slots = (unsigned int *) std::calloc((std::size_t) selected_count, sizeof(unsigned int));
    fleet->devices = (preprocess_workspace *) std::calloc((std::size_t) selected_count, sizeof(preprocess_workspace));
    fleet->results = (part_preprocess_result *) std::calloc((std::size_t) selected_count, sizeof(part_preprocess_result));
    if (fleet->slots == nullptr || fleet->devices == nullptr || fleet->results == nullptr) {
        clear(fleet);
        return 0;
    }
    fleet->slot_count = selected_count;
    for (unsigned int i = 0u; i < selected_count; ++i) init(fleet->devices + i);

    if (requested == 0u) {
        unsigned int default_count = cs_compute_runtime::default_mode_fleet_slots(fleet->fleet, fleet->slots, selected_count);
        if (default_count == 0u) default_count = cs_compute_runtime::default_generic_fleet_slots(fleet->fleet, fleet->slots, selected_count);
        fleet->slot_count = default_count;
    }

    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        const int requested_device = requested != 0u ? config->device_ids[i] : cs_compute_runtime::fleet_device_id(fleet->fleet, fleet->slots[i]);
        int found = -1;
        for (unsigned int slot = 0u; slot < fleet->fleet.local.device_count; ++slot) {
            if (fleet->fleet.local.device_ids[slot] == requested_device) {
                found = (int) slot;
                break;
            }
        }
        if (found < 0) {
            clear(fleet);
            return 0;
        }
        fleet->slots[i] = (unsigned int) found;
        if (!setup(fleet->devices + i, requested_device, cs_compute_runtime::fleet_stream(fleet->fleet, (unsigned int) found))) {
            clear(fleet);
            return 0;
        }
    }

#if CELLERATOR_DIST_HAS_NCCL
    if (config != nullptr && config->ranked_nccl != nullptr && config->ranked_nccl->unique_id != nullptr) {
        if (config->ranked_nccl->local_world_ranks == nullptr
            || config->ranked_nccl->world_size <= 0
            || config->ranked_nccl->unique_id_bytes != sizeof(ncclUniqueId)) {
            clear(fleet);
            return 0;
        }
        std::unique_ptr<int[]> device_ids(new (std::nothrow) int[selected_count]);
        if (!device_ids) {
            clear(fleet);
            return 0;
        }
        for (unsigned int i = 0u; i < selected_count; ++i) device_ids[i] = selected_device_id(fleet, i);
        ncclUniqueId unique_id;
        std::memcpy(&unique_id, config->ranked_nccl->unique_id, sizeof(unique_id));
        if (cs_dist::init_ranked_nccl_communicator(&fleet->ranked_nccl,
                                                   device_ids.get(),
                                                   fleet->slots,
                                                   selected_count,
                                                   config->ranked_nccl->local_world_ranks,
                                                   config->ranked_nccl->world_size,
                                                   &unique_id) != ncclSuccess) {
            clear(fleet);
            return 0;
        }
    }
#else
    if (config != nullptr && config->ranked_nccl != nullptr && config->ranked_nccl->unique_id != nullptr) {
        clear(fleet);
        return 0;
    }
#endif

    return 1;
}

} // namespace cellerator::compute::preprocess
