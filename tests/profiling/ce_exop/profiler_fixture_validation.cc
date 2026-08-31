#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
std::vector<std::string> split(std::string const& line)
{
    std::vector<std::string> result;
    std::stringstream input(line);
    for (std::string value; std::getline(input, value, '\t');)
    {
        result.push_back(value);
    }
    return result;
}

std::vector<std::vector<std::string>> read_tsv(char const* path, std::size_t columns)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("unable to open fixture");
    }
    std::vector<std::vector<std::string>> rows;
    for (std::string line; std::getline(input, line);)
    {
        auto fields = split(line);
        if (fields.size() != columns)
        {
            throw std::runtime_error("invalid fixture column count");
        }
        rows.push_back(std::move(fields));
    }
    return rows;
}
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        return 2;
    }
    auto fixtures = read_tsv(argv[1], 10);
    auto candidates = read_tsv(argv[2], 9);
    assert(fixtures.size() > 1 && candidates.size() > 1);

    std::set<std::string> fixture_ids;
    std::set<std::uint64_t> widths;
    std::set<std::uint64_t> segments;
    bool saw_high_degree = false;
    bool saw_mixed_cover = false;
    bool saw_fused = false;
    bool saw_projection_primary = false;
    for (std::size_t i = 1; i < fixtures.size(); ++i)
    {
        auto const& row = fixtures[i];
        assert(fixture_ids.insert(row[0]).second);
        widths.insert(std::stoull(row[2]));
        auto const degree = std::stoull(row[3]);
        auto const fraction = std::stod(row[4]);
        assert(fraction >= 0.0 && fraction <= 1.0);
        saw_high_degree |= degree > 4096;
        saw_mixed_cover |= fraction > 0.0 && fraction < 1.0;
        saw_fused |= row[8] != "unfused";
        saw_projection_primary |= row[5] == "projection_primary";
        if (row[1].find("segment_") == 0)
        {
            segments.insert(std::stoull(row[7]));
        }
    }
    for (std::uint64_t value : {15, 16, 17, 31, 32, 33, 64})
    {
        assert(widths.count(value) != 0);
    }
    for (std::uint64_t value : {1, 16, 17, 32, 33})
    {
        assert(segments.count(value) != 0);
    }
    assert(saw_high_degree && saw_mixed_cover && saw_fused && saw_projection_primary);

    std::set<std::string> candidate_ids;
    std::set<std::string> operations;
    std::set<std::string> fallback_operations;
    for (std::size_t i = 1; i < candidates.size(); ++i)
    {
        auto const& row = candidates[i];
        assert(candidate_ids.insert(row[0]).second);
        operations.insert(row[1]);
        assert(row[2] == "sm70");
        assert(std::stoull(row[4]) > 0);
        assert(row[5] == "true" || row[5] == "false");
        if (row[6] == "true")
        {
            fallback_operations.insert(row[1]);
        }
    }
    for (auto const& operation : operations)
    {
        assert(fallback_operations.count(operation) != 0);
    }
}
