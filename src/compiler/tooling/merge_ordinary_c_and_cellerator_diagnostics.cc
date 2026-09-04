#include <Cellerator/compiler/tooling/merge_ordinary_c_and_cellerator_diagnostics_v1.hh>
#include <algorithm>
namespace Cellerator::compiler::tooling {
std::vector<tooling_diagnostic_v1> merge_diagnostics_v1(std::vector<tooling_diagnostic_v1> clangd,
 std::vector<tooling_diagnostic_v1> cellerator,const diagnostic_remapper_v1 &remap) {
    std::vector<tooling_diagnostic_v1> out;
    for(auto &d:clangd){
        if(remap){ auto mapped=remap(d.range); if(!mapped) continue; d.range=*mapped;
            for(auto &f:d.fixes){ auto r=remap(f.range); if(r) f.range=*r; else f.replacement.clear(); }
            d.fixes.erase(std::remove_if(d.fixes.begin(),d.fixes.end(),[](const auto&f){return f.replacement.empty();}),d.fixes.end()); }
        out.push_back(std::move(d));
    }
    for(auto &d:cellerator){
        const auto duplicate=std::find_if(out.begin(),out.end(),[&](const auto &x){
            return x.range.begin==d.range.begin&&x.range.end==d.range.end&&x.code==d.code&&x.message==d.message;});
        if(duplicate==out.end()) out.push_back(std::move(d));
        else { duplicate->related.insert(duplicate->related.end(),d.related.begin(),d.related.end());
               duplicate->fixes.insert(duplicate->fixes.end(),d.fixes.begin(),d.fixes.end()); }
    }
    return out;
}
} // namespace Cellerator::compiler::tooling
