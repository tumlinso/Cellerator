#include <cassert>
struct node { int value; };
template<class P,class R> struct rule { P pattern; R rewrite; };
template<class P,class R> constexpr rule<P,R> make_rewrite(P p,R r) { return {p,r}; }
int main() {
    constexpr auto inline_rule = make_rewrite([](node n){return n.value==0;}, [](node){return node{1};});
    static_assert(inline_rule.pattern(node{0}));
    constexpr auto output = inline_rule.rewrite(node{0});
    assert(output.value == 1);
}
