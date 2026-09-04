#pragma once
#include <array>
#include <cstdint>
#include <string_view>
namespace Cellerator::compiler::migration {
struct temporary_adapter_v1{std::string_view legacy_surface,target_cellerator_contract,retirement_proof;std::uint32_t version;bool owns_semantics;};
inline constexpr std::array<temporary_adapter_v1,5> temporary_adapters_v1{{
 {"CellShard compiler evidence/discovery","Cellerator::compiler::profile","all evidence consumers compile against Cellerator profile APIs",1,false},
 {"CellShard compiler atom/certification","Cellerator::compiler::planning","all exact-certificate tests use Cellerator Planning IR",1,false},
 {"CellShard compiler grammar/basis/superatom","Cellerator::compiler::planning","all rule and promotion tests use public Cellerator extensions",1,false},
 {"CellShard compiler graph/schedule","Cellerator::compiler::program","all portable program and schedule consumers use Cellerator IR",1,false},
 {"CellShard compiler partial","Cellerator::compiler::partial","semantic tests move and only storage-consumer tests remain",1,false},
}};
} // namespace Cellerator::compiler::migration
