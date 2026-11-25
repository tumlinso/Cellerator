#include <Cellerator/memory/tiled/Tile.hh>

using cellerator::memory::Tile;

template<class T, unsigned Columns>
constexpr bool layout_matches() {
    return Tile<T>::rows == 16 && Tile<T>::columns == Columns && Tile<T>::row_bytes == 32
        && Tile<T>::bytes == 512 && sizeof(Tile<T>) == 512 && alignof(Tile<T>) == 32
        && __is_trivially_copyable(Tile<T>);
}

static_assert(layout_matches<unsigned char, 32>());
static_assert(layout_matches<unsigned short, 16>());
static_assert(layout_matches<float, 8>());
static_assert(layout_matches<double, 4>());

int main() {
    Tile<unsigned short> tiles[2];
    const auto &read = tiles;
    for (unsigned t = 0; t < 2; ++t)
        for (unsigned r = 0; r < 16; ++r)
            for (unsigned c = 0; c < 16; ++c)
                tiles[t].data()[r*16 + c] = static_cast<unsigned short>(t*256 + r*16 + c);

    const auto first = reinterpret_cast<__UINTPTR_TYPE__>(read[0].data());
    const auto second = reinterpret_cast<__UINTPTR_TYPE__>(read[1].data());
    if (first % 32 != 0 || second - first != 512) return 1;
    if (read[0].data()[16] != 16 || read[0].data()[255] != 255
        || read[1].data()[0] != 256 || read[1].data()[255] != 511) return 2;
}
