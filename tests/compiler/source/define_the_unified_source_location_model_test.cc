#include <Cellerator/compiler/frontend/source/define_the_unified_source_location_model_v1.hh>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace Cellerator::compiler::frontend::source;

int main() {
    try {
        source_map_v1 map;
        const std::string physical = "alpha\t\xce\xb2\r\ninclude\nlast";
        const auto file = map.add_space(source_space_kind_v1::physical_file,
                                        "src/example.ce.cc", physical);
        const auto include = map.add_space(source_space_kind_v1::include_instance,
                                           "src/header.hh@include:7", physical, file);
        if (file == invalid_source_space_v1 || include == invalid_source_space_v1) {
            throw std::runtime_error("source spaces were not created");
        }

        // Every byte boundary round-trips, including UTF-8 continuation bytes,
        // the two CRLF bytes, a tab, and the end-of-buffer location.
        for (std::uint64_t offset = 0; offset <= physical.size(); ++offset) {
            const auto position = map.line_column({file, offset});
            const auto round_trip = position ? map.location(file, *position) : std::nullopt;
            if (!round_trip || !(*round_trip == source_location_v1{file, offset})) {
                throw std::runtime_error("offset/line-column round trip failed");
            }
        }

        const std::array kinds{
            source_space_kind_v1::macro_expansion,
            source_space_kind_v1::transformed_buffer,
            source_space_kind_v1::ceir_node,
            source_space_kind_v1::backend_output,
        };
        source_space_id_v1 prior = include;
        for (std::size_t index = 0; index < kinds.size(); ++index) {
            const auto derived = map.add_space(kinds[index], "derived-" + std::to_string(index),
                                               physical, prior);
            const source_mapping_edge_v1 edge{
                {{derived, 0}, {derived, physical.size()}},
                {{prior, 0}, {prior, physical.size()}},
                index == 0 ? mapping_edge_kind_v1::macro_expansion
                           : mapping_edge_kind_v1::source_transform,
            };
            if (!map.add_mapping(edge)) {
                throw std::runtime_error("exact mapping edge was rejected");
            }
            const source_location_v1 point{derived, 8};
            const auto origin = map.map_to_origin(point);
            const auto round_trip = origin ? map.map_to_derived(*origin) : std::nullopt;
            if (!origin || !(*origin == source_location_v1{prior, 8}) ||
                !round_trip || !(*round_trip == point)) {
                throw std::runtime_error("mapping edge was not reversible");
            }
            prior = derived;
        }

        source_mapping_edge_v1 lossy{{{prior, 0}, {prior, 2}}, {{file, 0}, {file, 1}},
                                     mapping_edge_kind_v1::backend_provenance};
        if (map.add_mapping(lossy)) {
            throw std::runtime_error("lossy mapping edge was accepted");
        }

        std::cout << "validated reversible source locations across six source-space kinds\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
