#include <Cellerator/compiler/sema/implement_domain_and_human_biological_tag_semantics_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::sema::v1;
    biological_tag_registry tags;
    const auto gene = tags.find_tag("gene");
    const auto custom = tags.register_tag("organoid_compartment");
    assert(gene != no_biological_tag && custom != no_biological_tag);
    assert(tags.spelling(custom) == "organoid_compartment");

    domain_type a{{1, 2}, gene, false};
    domain_type b{{1, 2}, custom, false};
    domain_type c{{2, 1}, gene, false};
    assert(same_nominal_domain(a, b));
    assert(!same_nominal_domain(a, c));
    assert(same_nominal_domain(a, erase_domain_tag(a)));
    assert(can_explicitly_cast_domain(a, erase_nominal_domain(c), false));
    assert(!can_explicitly_cast_domain(a, c, false));
    assert(can_explicitly_cast_domain(a, c, true));

    domain_type abstract{{9, 4}, no_biological_tag, false};
    assert(same_nominal_domain(abstract, abstract));
}
