#include <Cellerator/sdk/freeze_libcellerator_component_architecture_v1.hh>

#include <cassert>
#include <set>

namespace ca = cellerator::compiler::api::v1;

int main() {
    assert(ca::component_link_graph_is_acyclic_v1());
    std::set<std::string_view> links;
    std::set<std::string_view> owners;
    for (std::uint32_t index = 0; index < ca::component_count_v1; ++index) {
        const auto contract = ca::component_contract(static_cast<ca::component_v1>(index));
        assert(!contract.link_name.empty() && !contract.abi_owner.empty());
        links.insert(contract.link_name);
        owners.insert(contract.abi_owner);
    }
    assert(links.size() == ca::component_count_v1);
    assert(owners.size() == ca::component_count_v1);
}
