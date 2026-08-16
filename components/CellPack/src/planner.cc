#include "CellPack/planner.hh"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace cellpack {
namespace {

struct module_builder {
    u32 module_id = invalid_id;
    std::vector<u32> features;
};

struct row_builder {
    u32 row = 0u;
    u64 hash = 0u;
    std::vector<u32> modules;
};

u64 fnv1a_modules(const std::vector<u32> &modules) {
    u64 hash = 1469598103934665603ull;
    for (u32 module : modules) {
        hash ^= static_cast<u64>(module);
        hash *= 1099511628211ull;
    }
    hash ^= static_cast<u64>(modules.size());
    hash *= 1099511628211ull;
    return hash;
}

bool same_signature(const row_builder &lhs, const row_builder &rhs) {
    return lhs.modules == rhs.modules;
}

u32 find_module_index(const std::vector<module_builder> &modules, u32 module_id) {
    for (u32 i = 0; i < static_cast<u32>(modules.size()); ++i) {
        if (modules[i].module_id == module_id) return i;
    }
    return invalid_id;
}

bool module_exists(const std::vector<module_builder> &modules, u32 module_id) {
    return find_module_index(modules, module_id) != invalid_id;
}

u32 resolve_residual_module_id(
    const feature_module_assignment_view &features,
    const planner_config &config) {
    if (config.residual_module_id != invalid_id) return config.residual_module_id;
    if (features.residual_module_id != invalid_id) return features.residual_module_id;
    return default_residual_module_id;
}

} // namespace

validation_result build_static_plan(
    const feature_module_assignment_view &features,
    const row_signature_view &rows,
    const planner_config &config,
    static_plan *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "static plan output is null");
    }
    if (features.feature_count != 0u && features.feature_to_module == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "feature module assignment is null");
    }
    if (rows.row_count != 0u && rows.row_offsets == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "row signature offsets are null");
    }
    if (rows.entry_count != 0u && rows.module_ids == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "row signature module ids are null");
    }
    if (rows.row_count == 0u && rows.entry_count != 0u) {
        return validation_error(validation_code::invalid_signature, invalid_id, "row signature entries require at least one row");
    }
    if (!is_valid_layout(config.primary_layout) || !is_valid_layout(config.residual_layout)) {
        return validation_error(validation_code::invalid_layout, invalid_id, "planner config contains an invalid layout");
    }
    if (config.residual_layout != layout_kind::residual_csr) {
        return validation_error(validation_code::invalid_layout, invalid_id, "M0/M1 residual regions must use residual CSR layout");
    }

    static_plan plan;
    const u32 residual_module_id = resolve_residual_module_id(features, config);
    plan.feature_block_offsets.push_back(0u);
    plan.row_group_offsets.push_back(0u);

    std::vector<module_builder> modules;
    module_builder residual_module;
    residual_module.module_id = residual_module_id;

    for (u32 feature = 0; feature < features.feature_count; ++feature) {
        const u32 module_id = features.feature_to_module[feature];
        if (module_id == invalid_id || module_id == residual_module_id) {
            residual_module.features.push_back(feature);
            continue;
        }
        u32 module_index = find_module_index(modules, module_id);
        if (module_index == invalid_id) {
            module_builder module;
            module.module_id = module_id;
            modules.push_back(std::move(module));
            module_index = static_cast<u32>(modules.size() - 1u);
        }
        modules[module_index].features.push_back(feature);
    }

    std::sort(modules.begin(), modules.end(), [](const module_builder &lhs, const module_builder &rhs) {
        return lhs.module_id < rhs.module_id;
    });
    for (module_builder &module : modules) {
        std::sort(module.features.begin(), module.features.end());
    }
    std::sort(residual_module.features.begin(), residual_module.features.end());

    plan.feature_permutation.reserve(features.feature_count);
    plan.modules.reserve(modules.size() + (residual_module.features.empty() ? 0u : 1u));
    for (const module_builder &module : modules) {
        feature_module_desc desc{};
        desc.module_id = module.module_id;
        desc.feature_begin = static_cast<u32>(plan.feature_permutation.size());
        desc.feature_count = static_cast<u32>(module.features.size());
        desc.flags = module_flag_none;
        plan.modules.push_back(desc);
        plan.feature_permutation.insert(plan.feature_permutation.end(), module.features.begin(), module.features.end());
        plan.feature_block_offsets.push_back(static_cast<u32>(plan.feature_permutation.size()));
    }
    const u32 residual_feature_begin = static_cast<u32>(plan.feature_permutation.size());
    if (!residual_module.features.empty()) {
        feature_module_desc desc{};
        desc.module_id = residual_module.module_id;
        desc.feature_begin = residual_feature_begin;
        desc.feature_count = static_cast<u32>(residual_module.features.size());
        desc.flags = module_flag_residual;
        plan.modules.push_back(desc);
        plan.feature_permutation.insert(
            plan.feature_permutation.end(),
            residual_module.features.begin(),
            residual_module.features.end());
        plan.feature_block_offsets.push_back(static_cast<u32>(plan.feature_permutation.size()));
    }

    plan.inverse_feature_permutation.resize(features.feature_count);
    if (!build_inverse_permutation(
            plan.feature_permutation.data(),
            static_cast<u32>(plan.feature_permutation.size()),
            plan.inverse_feature_permutation.data())) {
        return validation_error(validation_code::invalid_permutation, invalid_id, "feature permutation is not invertible");
    }

    if (rows.row_count != 0u && rows.row_offsets[0] != 0u) {
        return validation_error(validation_code::invalid_signature, 0u, "row signature offsets must start at zero");
    }
    std::vector<row_builder> row_builders;
    row_builders.reserve(rows.row_count);
    for (u32 row = 0; row < rows.row_count; ++row) {
        const u32 begin = rows.row_offsets[row];
        const u32 end = rows.row_offsets[row + 1u];
        if (end < begin || end > rows.entry_count) {
            return validation_error(validation_code::invalid_signature, row, "row signature offsets are not monotonic");
        }
        row_builder builder;
        builder.row = row;
        builder.modules.reserve(end - begin);
        for (u32 entry = begin; entry < end; ++entry) {
            const u32 module_id = rows.module_ids[entry];
            if (module_id == invalid_id || module_id == residual_module_id) continue;
            if (!module_exists(modules, module_id)) {
                return validation_error(validation_code::unknown_module, row, "row signature references a module with no assigned features");
            }
            builder.modules.push_back(module_id);
        }
        std::sort(builder.modules.begin(), builder.modules.end());
        builder.modules.erase(std::unique(builder.modules.begin(), builder.modules.end()), builder.modules.end());
        builder.hash = fnv1a_modules(builder.modules);
        row_builders.push_back(std::move(builder));
    }
    std::sort(row_builders.begin(), row_builders.end(), [](const row_builder &lhs, const row_builder &rhs) {
        if (lhs.modules != rhs.modules) return lhs.modules < rhs.modules;
        return lhs.row < rhs.row;
    });

    plan.row_permutation.reserve(rows.row_count);
    for (const row_builder &row : row_builders) plan.row_permutation.push_back(row.row);
    plan.inverse_row_permutation.resize(rows.row_count);
    if (!build_inverse_permutation(
            plan.row_permutation.data(),
            static_cast<u32>(plan.row_permutation.size()),
            plan.inverse_row_permutation.data())) {
        return validation_error(validation_code::invalid_permutation, invalid_id, "row permutation is not invertible");
    }

    plan.signature_offsets.push_back(0u);
    u32 group_begin = 0u;
    while (group_begin < static_cast<u32>(row_builders.size())) {
        u32 group_end = group_begin + 1u;
        while (group_end < static_cast<u32>(row_builders.size())
               && same_signature(row_builders[group_begin], row_builders[group_end])) {
            ++group_end;
        }

        const row_builder &signature = row_builders[group_begin];
        const u32 signature_offset = static_cast<u32>(plan.signature_module_ids.size());
        plan.signature_module_ids.insert(
            plan.signature_module_ids.end(),
            signature.modules.begin(),
            signature.modules.end());
        plan.signature_offsets.push_back(static_cast<u32>(plan.signature_module_ids.size()));

        row_group_desc row_group{};
        row_group.signature_hash = signature.hash;
        row_group.row_begin = group_begin;
        row_group.row_count = group_end - group_begin;
        row_group.signature_offset = signature_offset;
        row_group.signature_count = static_cast<u32>(signature.modules.size());
        row_group.flags = 0u;
        plan.row_groups.push_back(row_group);
        plan.row_group_offsets.push_back(group_end);

        const region_role role = row_group.row_count < config.min_primary_rows
            ? region_role::conditional
            : region_role::primary;
        const u32 flags = role == region_role::conditional ? region_flag_conditional : region_flag_none;
        for (u32 signature_index = 0; signature_index < static_cast<u32>(signature.modules.size()); ++signature_index) {
            const u32 module_id = signature.modules[signature_index];
            const u32 module_index = find_module_index(modules, module_id);
            const feature_module_desc &module_desc = plan.modules[module_index];

            packed_region_desc region{};
            region.region_id = static_cast<u32>(plan.regions.size());
            region.parent_id = invalid_id;
            region.flags = flags;
            region.layout = to_u32(config.primary_layout);
            region.role = to_u32(role);
            region.module_id = module_id;
            region.signature_id = static_cast<u32>(plan.row_groups.size() - 1u);
            region.row_begin = row_group.row_begin;
            region.row_count = row_group.row_count;
            region.feature_begin = module_desc.feature_begin;
            region.feature_count = module_desc.feature_count;
            region.block_size = 0u;
            region.width_class = module_desc.feature_count;
            region.index_offset = invalid_id;
            region.value_offset = invalid_id;
            region.aux_offset = invalid_id;
            region.weight_offset = invalid_id;
            region.output_offset = invalid_id;
            region.nnz_count = 0u;
            plan.regions.push_back(region);
        }

        group_begin = group_end;
    }

    if (config.emit_residual_region && !residual_module.features.empty()) {
        packed_region_desc residual{};
        residual.region_id = static_cast<u32>(plan.regions.size());
        residual.parent_id = invalid_id;
        residual.flags = region_flag_residual;
        residual.layout = to_u32(layout_kind::residual_csr);
        residual.role = to_u32(region_role::residual);
        residual.module_id = residual_module_id;
        residual.signature_id = invalid_id;
        residual.row_begin = 0u;
        residual.row_count = rows.row_count;
        residual.feature_begin = residual_feature_begin;
        residual.feature_count = static_cast<u32>(residual_module.features.size());
        residual.block_size = 0u;
        residual.width_class = 0u;
        residual.index_offset = invalid_id;
        residual.value_offset = invalid_id;
        residual.aux_offset = invalid_id;
        residual.weight_offset = invalid_id;
        residual.output_offset = invalid_id;
        residual.nnz_count = 0u;
        plan.regions.push_back(residual);
    }

    plan.desc.version = abi_version;
    plan.desc.flags = 0u;
    plan.desc.row_count = rows.row_count;
    plan.desc.feature_count = features.feature_count;
    plan.desc.row_permutation.count = rows.row_count;
    plan.desc.row_permutation.permutation_offset = 0u;
    plan.desc.row_permutation.inverse_offset = rows.row_count;
    plan.desc.row_permutation.flags = is_identity_permutation(plan.row_permutation.data(), rows.row_count)
        ? permutation_flag_identity
        : permutation_flag_none;
    plan.desc.feature_permutation.count = features.feature_count;
    plan.desc.feature_permutation.permutation_offset = 0u;
    plan.desc.feature_permutation.inverse_offset = features.feature_count;
    plan.desc.feature_permutation.flags = is_identity_permutation(plan.feature_permutation.data(), features.feature_count)
        ? permutation_flag_identity
        : permutation_flag_none;
    plan.desc.module_count = static_cast<u32>(plan.modules.size());
    plan.desc.row_group_count = static_cast<u32>(plan.row_groups.size());
    plan.desc.region_count = static_cast<u32>(plan.regions.size());
    plan.desc.residual_region_count = (!residual_module.features.empty() && config.emit_residual_region) ? 1u : 0u;
    plan.desc.module_desc_offset = 0u;
    plan.desc.row_group_desc_offset = 0u;
    plan.desc.region_desc_offset = 0u;
    plan.desc.signature_offset = 0u;

    validation_result desc_result = validate_plan_desc(plan.desc);
    if (!desc_result) return desc_result;
    validation_result region_result = validate_region_sequence(
        plan.regions.data(),
        static_cast<u32>(plan.regions.size()),
        rows.row_count,
        features.feature_count);
    if (!region_result) return region_result;

    *out = std::move(plan);
    return validation_ok();
}

} // namespace cellpack
