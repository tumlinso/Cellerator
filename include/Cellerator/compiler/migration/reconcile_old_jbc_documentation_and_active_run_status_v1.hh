#pragma once
#include <array>
#include <string_view>
namespace Cellerator::compiler::migration {
struct jbc_supersession_v1{std::string_view historical_class,replacement_authority;bool preserve_record;bool additive_only;};
inline constexpr std::array<jbc_supersession_v1,4> jbc_supersession_v1{{
 {"active historical JBC run","CE-CCP1-RUN-V1",true,true},{"JBC charters","Cellerator architecture and Part One contracts",true,true},{"JBC interfaces","versioned Cellerator interfaces and compatibility readers",true,true},{"JBC package documents","Cellerator compiler package and migration receipts",true,true},
}};
[[nodiscard]] constexpr bool preserves_history_v1()noexcept{for(auto r:jbc_supersession_v1)if(!r.preserve_record||!r.additive_only||r.replacement_authority.empty())return false;return true;}
} // namespace Cellerator::compiler::migration
