#include "bench/benchmark_mutex.hh"
#include <array>
#include <iostream>
#include <string_view>
struct measurement { std::string_view corpus, baseline; unsigned repetitions; bool wall, rss, depfile, object_size, diagnostics; };
int main() {
    cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-j02-plain-c");
    constexpr std::array<std::string_view,3> corpora{"small","medium","template-heavy"};
    constexpr std::array<std::string_view,2> baselines{"direct-gcc","direct-clang"};
    for(auto corpus:corpora) for(auto baseline:baselines) {
        constexpr unsigned repeats=11;
        measurement row{corpus,baseline,repeats,true,true,true,true,true};
        std::cout<<row.corpus<<','<<row.baseline<<','<<row.repetitions<<",capture-contract\n";
    }
}
