#include <cassert>
#include <set>
#include <string>
#include <vector>
struct negative_case{std::string source,id;bool unsafe=false;};
int main(){const std::vector<negative_case>cases={{"state<cell,gene> == state<sample,gene>","CE-SEMA-DOMAIN-IDENTITY"},{"relation<gene,cell> applied_to protein","CE-SEMA-ENDPOINT"},{"canonical consumed as packed","CE-SEMA-ORDER"},{"values generation 7 bound to 8","CE-SEMA-GENERATION"},{"forward relation used as transpose","CE-SEMA-ORIENTATION"},{"stale values after mutation","CE-SEMA-STALE-VALUE"},{"overwrite and accumulate","CE-SEMA-EFFECT"},{"fp8 input with exact fp64 accumulation","CE-SEMA-NUMERIC-TUPLE"},{"reinterpret_domain without unsafe","CE-SEMA-UNSAFE-REQUIRED"},{"unsafe changes relation meaning","CE-SEMA-UNSAFE-CORRECTNESS",true}};std::set<std::string>ids;for(const auto&c:cases){assert(!c.source.empty()&&c.id.rfind("CE-SEMA-",0)==0);assert(ids.insert(c.id).second);}assert(cases.size()==10&&cases.back().unsafe);}
