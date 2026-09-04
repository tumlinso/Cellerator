#include <cassert>
#include <cstddef>
#include <type_traits>

template <class T> struct view { T* data; std::size_t count; };
enum class status_code : unsigned { ok, invalid_argument, out_of_capacity };
template <class T> struct result { T value; status_code status; };

int main() {
    static_assert(std::is_trivially_copyable_v<view<int>>);
    static_assert(std::is_trivially_copyable_v<result<int>>);
    int values[]{2, 3};
    view<int> input{values, 2};
    assert(input.data[0] + input.data[1] == 5);
    const result<int> output{5, status_code::ok};
    assert(output.status == status_code::ok);
}
