#include <Cellerator/compiler/lto/implement_elf_ceir_sections_v1.hh>
#include <cassert>
using namespace cellerator::compiler::lto::v1;
int main(){elf_ceir_section_v1 s{".cellerator.ceir.v1","schema=1","__cellerator_ceir",{1,2,3},elf_ceir_compression_v1::zstd,false,true};auto a=emit_elf_ceir_section_v1(s),b=emit_elf_ceir_section_v1(s);assert(a==b&&!a.empty());auto x=extract_elf_ceir_section_v1(a);assert(x&&x->payload==s.payload&&!x->allocatable);assert(strip_elf_runtime_symbols_v1(a)==a);s.retain_when_stripped=false;assert(strip_elf_runtime_symbols_v1(emit_elf_ceir_section_v1(s)).empty());}
