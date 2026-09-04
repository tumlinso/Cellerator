#include <Cellerator/compiler/ir/realization/commit_selected_exact_cover_and_contribution_ownership_v1.hh>

#include <algorithm>
#include <set>
#include <tuple>

namespace cellerator::compiler::ir::realization::v1 {
namespace {

using identity_key_v1 = std::tuple<std::uint64_t, std::uint64_t>;

identity_key_v1 key(stable_identity_v1 identity) noexcept {
    return {identity.high, identity.low};
}

exact_cover_status_v1 fail(
    exact_cover_status_v1 status, std::string* error, const char* message) noexcept {
    if (error != nullptr) {
        *error = message;
    }
    return status;
}

} // namespace

exact_cover_status_v1 validate_exact_cover_v1(
    const exact_cover_v1& cover, std::string* error) noexcept {
    if (!valid(cover.identity)) {
        return fail(exact_cover_status_v1::invalid_identity, error,
            "exact-cover identity is required");
    }
    if (!valid(cover.certification_receipt)) {
        return fail(exact_cover_status_v1::invalid_receipt, error,
            "certification receipt is required");
    }
    std::vector<bool> seen(cover.logical_item_count, false);
    std::vector<bool> recovered(cover.logical_item_count, false);
    for (const auto& entry : cover.entries) {
        if (entry.logical_item >= cover.logical_item_count) {
            return fail(exact_cover_status_v1::omitted_item, error,
                "logical item lies outside the certified universe");
        }
        if (seen[entry.logical_item]) {
            return fail(exact_cover_status_v1::duplicate_item, error,
                "logical item has more than one exact-cover entry");
        }
        seen[entry.logical_item] = true;
        if (!valid(entry.atom) || !valid(entry.owner)) {
            return fail(exact_cover_status_v1::invalid_owner, error,
                "atom and contribution owner are required");
        }
        std::set<identity_key_v1> placements{key(entry.owner)};
        for (const auto replica : entry.replicas) {
            if (!valid(replica) || !placements.insert(key(replica)).second) {
                return fail(exact_cover_status_v1::duplicate_replica, error,
                    "replicas must be valid, unique, and distinct from the owner");
            }
        }
        for (const auto halo : entry.halos) {
            if (!valid(halo)) {
                return fail(exact_cover_status_v1::invalid_owner, error,
                    "halo identities are required");
            }
        }
        std::set<identity_key_v1> contributors;
        for (const auto& contributor : entry.contributors) {
            if (!valid(contributor.identity) || contributor.denominator <= 0 ||
                contributor.numerator == 0 ||
                !contributors.insert(key(contributor.identity)).second) {
                return fail(exact_cover_status_v1::invalid_contributor, error,
                    "partial contributors require unique identities and exact nonzero weights");
            }
        }
        if (entry.canonical_recovery >= cover.logical_item_count ||
            recovered[entry.canonical_recovery]) {
            return fail(exact_cover_status_v1::invalid_recovery, error,
                "canonical recovery must be a permutation");
        }
        recovered[entry.canonical_recovery] = true;
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        return fail(exact_cover_status_v1::omitted_item, error,
            "certified logical item is omitted");
    }
    if (std::find(recovered.begin(), recovered.end(), false) != recovered.end()) {
        return fail(exact_cover_status_v1::invalid_recovery, error,
            "canonical recovery is incomplete");
    }
    if (error != nullptr) {
        error->clear();
    }
    return exact_cover_status_v1::exact;
}

exact_cover_status_v1 validate_exact_cover_rewrite_v1(
    const exact_cover_v1& before,
    const exact_cover_v1& after,
    std::string* error) noexcept {
    auto status = validate_exact_cover_v1(before, error);
    if (status != exact_cover_status_v1::exact) {
        return status;
    }
    status = validate_exact_cover_v1(after, error);
    if (status != exact_cover_status_v1::exact) {
        return status;
    }
    if (before.logical_item_count != after.logical_item_count ||
        !(before.certification_receipt == after.certification_receipt)) {
        return fail(exact_cover_status_v1::rewrite_changed_cover, error,
            "rewrite changed the certified universe or receipt");
    }
    for (std::uint64_t item = 0u; item < before.logical_item_count; ++item) {
        const auto find_item = [item](const exact_cover_entry_v1& entry) {
            return entry.logical_item == item;
        };
        const auto a = std::find_if(before.entries.begin(), before.entries.end(), find_item);
        const auto b = std::find_if(after.entries.begin(), after.entries.end(), find_item);
        if (!(a->atom == b->atom) || a->canonical_recovery != b->canonical_recovery) {
            return fail(exact_cover_status_v1::rewrite_changed_cover, error,
                "rewrite changed atom coverage or canonical recovery");
        }
    }
    if (error != nullptr) {
        error->clear();
    }
    return exact_cover_status_v1::exact;
}

} // namespace cellerator::compiler::ir::realization::v1
