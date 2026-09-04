#include <cassert>
#include <cstddef>
struct identity { unsigned long long value; };
struct generation { unsigned long long value; };
template<class T> struct state_view { T* data; std::size_t extent; identity domain; identity order; generation values; };
template<class T> constexpr state_view<T> make_state_view(T* p, std::size_t n, identity d, identity o, generation g) { return {p,n,d,o,g}; }
int main() {
    float values[2]{};
    const auto view = make_state_view(values, 2, identity{3}, identity{7}, generation{11});
    assert(view.data == values && view.extent == 2 && view.domain.value == 3 && view.order.value == 7 && view.values.value == 11);
}
