#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/accessibility.hh>

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    using namespace cellerator::modeling;
    using Program = AccessibilityProgram<std::string>;
    AccessibilityState<std::string> founder({
        Program{"express", {"accessible", "signal"}, {"repressor"}, 4, true},
        Program{"quiet", {"accessible"}, {}, 1, false},
    }, 7);
    assert(founder.structure_revision() == 7);
    assert(founder.program_count() == 2 && founder.input_count() == 3);
    assert(!founder.eligible("express"));
    founder.set_input("accessible", true, "inherited_chromatin");
    assert(!founder.eligible("express"));
    founder.set_inputs({{"signal", true}}, "external_signal");
    assert(founder.eligible("express"));
    assert(founder.query("express").program_revision == 4);
    founder.set_input("repressor", true, "state_update");
    assert(!founder.eligible("express"));

    auto daughter_a = founder.fork();
    auto daughter_b = founder.fork();
    assert(founder.shares_structure_with(daughter_a));
    assert(daughter_a.shares_structure_with(daughter_b));
    daughter_a.set_input("repressor", false, "division_inheritance");
    daughter_a.set_open("quiet", true, "developmental_program");
    assert(daughter_a.eligible("express"));
    assert(!daughter_b.eligible("express"));
    assert(!founder.eligible("quiet") && daughter_a.eligible("quiet"));
    assert(founder.generation() != daughter_a.generation());
    assert(daughter_a.changes().back().provenance == "developmental_program");
    assert(daughter_a.query("express").structure_revision == 7);

    bool rejected = false;
    try { founder.set_inputs({{"accessible", false}, {"not_known", true}}); }
    catch (const std::out_of_range &) { rejected = true; }
    assert(rejected && founder.input_active("accessible"));

    const HysteresisThreshold threshold(0.8, 0.2);
    assert(threshold.update(false, 0.81));
    assert(threshold.update(true, 0.21));
    assert(!threshold.update(true, 0.19));
    assert(!threshold.update(false, 0.79));
    assert(founder.storage_bytes() == sizeof(std::uint64_t) * 2);
    return 0;
}
