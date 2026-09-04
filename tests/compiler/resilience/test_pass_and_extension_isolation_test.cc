#include <cassert>
#include <string>
#include <vector>
enum class trust{isolated,verified,trusted};struct failure{std::string kind;trust mode;bool continue_compilation;};
int main(){std::vector<failure>x={{"invalid-ir",trust::isolated,true},{"unknown-extension",trust::verified,false},{"crash",trust::isolated,true},{"timeout",trust::isolated,true},{"recursive-transform",trust::trusted,false},{"stale-plugin",trust::verified,false},{"false-preservation",trust::trusted,false}};for(auto f:x){if(f.mode==trust::isolated)assert(f.continue_compilation);else assert(!f.continue_compilation);} }
