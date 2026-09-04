#include <Cellerator/compiler/ir/realization/commit_selected_exact_cover_and_contribution_ownership_v1.hh>

#include <algorithm>
#include <cassert>

using namespace cellerator::compiler::ir::realization::v1;

int main() {
    exact_cover_v1 cover;
    cover.identity = {1u, 1u};
    cover.certification_receipt = {1u, 2u};
    cover.logical_item_count = 3u;
    cover.entries = {
        {0u, {2u, 1u}, {3u, 1u}, {{4u, 1u}}, {{5u, 1u}}, {{{6u, 1u}, 1, 2}}, 2u},
        {1u, {2u, 2u}, {3u, 1u}, {}, {}, {{{6u, 2u}, 1, 1}}, 0u},
        {2u, {2u, 3u}, {3u, 2u}, {}, {}, {}, 1u},
    };
    assert(validate_exact_cover_v1(cover) == exact_cover_status_v1::exact);

    auto rewritten = cover;
    std::reverse(rewritten.entries.begin(), rewritten.entries.end());
    rewritten.entries.front().owner = {3u, 3u};
    assert(validate_exact_cover_rewrite_v1(cover, rewritten) ==
        exact_cover_status_v1::exact);

    auto duplicate = cover;
    duplicate.entries[1].logical_item = 0u;
    assert(validate_exact_cover_v1(duplicate) == exact_cover_status_v1::duplicate_item);

    auto omitted = cover;
    omitted.entries.pop_back();
    assert(validate_exact_cover_v1(omitted) == exact_cover_status_v1::omitted_item);

    auto changed = rewritten;
    changed.entries.front().atom = {9u, 9u};
    assert(validate_exact_cover_rewrite_v1(cover, changed) ==
        exact_cover_status_v1::rewrite_changed_cover);
}
