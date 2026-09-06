#pragma once
#include "oracle.hh"
#include <array>
namespace spine_verify {
inline constexpr std::array<edge, 9> demo = {{{0,0,0x3800},{2,0,0xb400},{1,1,0x3c00},
    {3,1,0x3000},{0,2,0xb800},{3,2,0x3a00},{0,4,0x3400},{1,4,0xbc00},{2,4,0x3800}}};
inline constexpr std::array<edge, 9> demo_generation_2 = {{{0,0,0x3c00},{2,0,0xb400},{1,1,0x3800},
    {3,1,0x3000},{0,2,0xb800},{3,2,0x3400},{0,4,0x3400},{1,4,0xbc00},{2,4,0x3800}}};
inline constexpr float demo_input[] = {2,1,4,2}, demo_signal[] = {1,-2,.5,3,2};
// Three sources, four destinations, empty row 1; endpoint duplicates are distinct edges.
inline constexpr std::array<edge, 5> duplicate = {{{2,0,0x3c00},{0,2,0x3800},
    {0,2,0x3400},{1,3,0xc000},{2,3,0x3c00}}};
inline constexpr float second_input[] = {4,2,3}, second_signal[] = {2,17,-2,1};
inline constexpr std::array<edge, 4> second = {{{2,0,0x3c00},{0,2,0x3800},
    {1,3,0xc000},{2,3,0x3c00}}};
inline constexpr std::array<edge, 3> cancellation = {{{0,0,0x7bff},{1,0,0xfbff},{2,0,0x3c00}}};
}
