#include <Cellerator/compiler/ast/assign_deterministic_source_identities_v1.hh>

#include <set>

namespace Cellerator::compiler::ast {
namespace {

constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void hash_word(std::uint64_t word, std::uint64_t* hash) noexcept {
    for (unsigned byte = 0; byte != 8; ++byte) {
        *hash ^= (word >> (byte * 8U)) & 0xffU;
        *hash *= fnv_prime;
    }
}

} // namespace

compilation_source_identity_v1 derive_source_identity_v1(
    const source_identity_input_v1& input) noexcept {
    std::uint64_t high = 14695981039346656037ULL;
    std::uint64_t low = 7809847782465536322ULL;
    const std::uint64_t words[] = {
        0x43454c4c45524131ULL, input.semantic_owner, input.canonical_file,
        input.canonical_offset, input.declaration_identity, input.language_revision};
    for (const auto word : words) {
        hash_word(word, &high);
        hash_word(word ^ 0x9e3779b97f4a7c15ULL, &low);
    }
    // Reserve the all-zero value for invalid/unassigned identities.
    if ((high | low) == 0) low = 1;
    return {high, low};
}

std::optional<std::vector<source_identity_record_v1>>
assign_source_identities_v1(const std::vector<source_identity_input_v1>& inputs,
                            const std::vector<ast_node_handle_v1>& transient_nodes,
                            const std::vector<std::optional<persistent_user_identity_v1>>&
                                persistent_identities,
                            std::string* error) {
    auto fail = [&](std::string message)
        -> std::optional<std::vector<source_identity_record_v1>> {
        if (error) *error = std::move(message);
        return std::nullopt;
    };
    if (inputs.size() != transient_nodes.size() ||
        inputs.size() != persistent_identities.size())
        return fail("identity inputs, AST handles, and persistent identities must align");

    std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
    std::vector<source_identity_record_v1> records;
    records.reserve(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto& input = inputs[index];
        if (input.semantic_owner == 0 || input.canonical_file == 0 ||
            input.declaration_identity == 0 || input.language_revision == 0 ||
            !transient_nodes[index].valid())
            return fail("source identity input is incomplete");
        const auto identity = derive_source_identity_v1(input);
        if (!seen.emplace(identity.high, identity.low).second)
            return fail("duplicate source identity input");
        records.push_back({identity, transient_nodes[index], persistent_identities[index]});
    }
    if (error) error->clear();
    return records;
}

} // namespace Cellerator::compiler::ast
