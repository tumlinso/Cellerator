#include <Cellerator/compiler/ast/support_incremental_ast_identity_reuse_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::ast {
namespace {

bool same_locator(const incremental_ast_identity_v1& left,
                  const incremental_ast_identity_v1& right) noexcept {
    return left.level == right.level && left.stable_locator == right.stable_locator;
}

bool reusable(const incremental_ast_identity_v1& previous,
              const incremental_ast_identity_v1& current) noexcept {
    if (previous.content_hash != current.content_hash ||
        previous.dependency_hash != current.dependency_hash ||
        previous.macro_dependent != current.macro_dependent ||
        previous.template_dependent != current.template_dependent)
        return false;
    if (current.macro_dependent &&
        (current.macro_environment_hash == 0 ||
         previous.macro_environment_hash != current.macro_environment_hash))
        return false;
    if (current.template_dependent &&
        (current.template_environment_hash == 0 ||
         previous.template_environment_hash != current.template_environment_hash))
        return false;
    return true;
}

} // namespace

std::optional<incremental_reuse_result_v1>
reuse_incremental_ast_identities_v1(
    std::vector<incremental_ast_identity_v1> previous,
    std::vector<incremental_ast_identity_v1> current,
    std::string* error) {
    const auto fail = [&](std::string message) -> std::optional<incremental_reuse_result_v1> {
        if (error) *error = std::move(message);
        return std::nullopt;
    };
    const auto order = [](const auto& left, const auto& right) {
        return left.level < right.level ||
               (left.level == right.level && left.stable_locator < right.stable_locator);
    };
    std::sort(previous.begin(), previous.end(), order);
    std::sort(current.begin(), current.end(), order);
    std::uint64_t next_identity = 1;
    for (std::size_t index = 0; index < previous.size(); ++index) {
        if (previous[index].identity == 0 || previous[index].stable_locator == 0 ||
            previous[index].content_hash == 0 || previous[index].dependency_hash == 0 ||
            (index && same_locator(previous[index - 1], previous[index])))
            return fail("previous incremental identity set is invalid");
        next_identity = std::max(next_identity, previous[index].identity + 1);
    }
    for (std::size_t index = 0; index < current.size(); ++index)
        if (current[index].identity != 0 || current[index].stable_locator == 0 ||
            current[index].content_hash == 0 || current[index].dependency_hash == 0 ||
            (index && same_locator(current[index - 1], current[index])))
            return fail("current incremental fingerprint set is invalid");

    incremental_reuse_result_v1 result;
    result.metrics.total = current.size();
    for (auto& item : current) {
        const auto found = std::lower_bound(previous.begin(), previous.end(), item, order);
        if (found != previous.end() && same_locator(*found, item) && reusable(*found, item)) {
            item.identity = found->identity;
            ++result.metrics.reused;
        } else {
            if (found != previous.end() && same_locator(*found, item)) ++result.metrics.invalidated;
            item.identity = next_identity++;
            ++result.metrics.created;
        }
    }
    result.identities = std::move(current);
    if (error) error->clear();
    return result;
}

} // namespace Cellerator::compiler::ast
