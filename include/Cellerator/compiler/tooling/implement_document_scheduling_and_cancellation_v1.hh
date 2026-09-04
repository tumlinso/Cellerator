#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace Cellerator::compiler::tooling {

enum class document_work_kind_v1 : std::uint8_t { parse = 1, profile, plan };

struct document_work_v1 {
    std::string uri;
    std::uint64_t generation = 0;
    std::uint64_t ready_at_ms = 0;
    document_work_kind_v1 kind = document_work_kind_v1::parse;
    bool active = false;
};

class document_scheduler_v1 {
public:
    explicit document_scheduler_v1(std::uint64_t debounce_ms = 50,
                                   std::size_t maximum_background = 4);
    void edited(std::string uri, std::uint64_t generation, std::uint64_t now_ms, bool active);
    void request_slow(std::string uri, std::uint64_t generation, document_work_kind_v1 kind);
    [[nodiscard]] std::optional<document_work_v1> next_basic(std::uint64_t now_ms);
    [[nodiscard]] std::optional<document_work_v1> next_slow();
    [[nodiscard]] bool cancelled(const document_work_v1 &work) const noexcept;
    [[nodiscard]] std::size_t pending_basic() const noexcept { return basic_.size(); }
    [[nodiscard]] std::size_t pending_slow() const noexcept { return slow_.size(); }

private:
    std::uint64_t debounce_ms_;
    std::size_t maximum_background_;
    std::unordered_map<std::string, std::uint64_t> generations_;
    std::vector<document_work_v1> basic_;
    std::vector<document_work_v1> slow_;
};

} // namespace Cellerator::compiler::tooling
