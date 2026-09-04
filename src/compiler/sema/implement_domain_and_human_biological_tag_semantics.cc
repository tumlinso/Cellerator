#include <Cellerator/compiler/sema/implement_domain_and_human_biological_tag_semantics_v1.hh>

#include <array>

namespace cellerator::compiler::sema::v1 {

biological_tag_registry::biological_tag_registry() {
    constexpr std::array<const char *, 9> common{{
        "gene", "cell", "locus", "enhancer", "read", "chromosome",
        "population", "trajectory", "module"}};
    for (const char *tag : common)
        register_tag(tag);
}

biological_tag_id biological_tag_registry::register_tag(std::string spelling) {
    if (spelling.empty())
        return no_biological_tag;
    const auto found = ids_.find(spelling);
    if (found != ids_.end())
        return found->second;
    const biological_tag_id id = next_id_++;
    spellings_.emplace(id, spelling);
    ids_.emplace(std::move(spelling), id);
    return id;
}

biological_tag_id biological_tag_registry::find_tag(
    std::string_view spelling) const noexcept {
    const auto found = ids_.find(std::string(spelling));
    return found == ids_.end() ? no_biological_tag : found->second;
}

std::string_view biological_tag_registry::spelling(
    biological_tag_id tag) const noexcept {
    const auto found = spellings_.find(tag);
    return found == spellings_.end() ? std::string_view{} : found->second;
}

bool same_nominal_domain(const domain_type &left,
                         const domain_type &right) noexcept {
    if (left.domain_erased || right.domain_erased)
        return left.domain_erased && right.domain_erased;
    return left.identity.low == right.identity.low
        && left.identity.high == right.identity.high;
}

domain_type erase_domain_tag(domain_type domain) noexcept {
    domain.diagnostic_tag = no_biological_tag;
    return domain;
}

domain_type erase_nominal_domain(domain_type domain) noexcept {
    domain.identity = {};
    domain.diagnostic_tag = no_biological_tag;
    domain.domain_erased = true;
    return domain;
}

bool can_explicitly_cast_domain(const domain_type &source,
                                const domain_type &destination,
                                bool unsafe_authorized) noexcept {
    return same_nominal_domain(source, destination)
        || source.domain_erased || destination.domain_erased
        || unsafe_authorized;
}

}  // namespace cellerator::compiler::sema::v1
