#include <Cellerator/compiler/diagnostics/implement_deterministic_reproducer_bundles_v1.hh>
#include <algorithm>
#include <cassert>
int main(){using namespace cellerator::compiler::diagnostics::v1;std::vector<bundle_entry> a{{bundle_entry_kind::diagnostic,"missing operand"},{bundle_entry_kind::source_subset,"x.cell"},{bundle_entry_kind::command,"cellerator -c x.cell"},{bundle_entry_kind::profile,"p"},{bundle_entry_kind::ceir_checkpoint,"ir"},{bundle_entry_kind::toolchain_manifest,"v1"},{bundle_entry_kind::pipeline,"default"},{bundle_entry_kind::extension,"none"}};auto b=make_reproducer_bundle(a);std::reverse(a.begin(),a.end());auto c=make_reproducer_bundle(a);assert(b.digest==c.digest&&replay_matches(b,c.digest));b.contains_dataset_payload=true;assert(!replay_matches(b,c.digest));}
