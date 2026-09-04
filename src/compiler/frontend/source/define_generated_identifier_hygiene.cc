#include <Cellerator/compiler/frontend/source/define_generated_identifier_hygiene_v1.hh>

#include <array>
#include <iomanip>
#include <sstream>

namespace Cellerator::compiler::frontend::source {

generated_identifier_v1 make_generated_identifier_v1(generated_identifier_domain_v1 domain,
                                                       std::string_view content) {
    std::uint64_t hash = 1469598103934665603ULL ^ static_cast<std::uint8_t>(domain);
    for (unsigned char byte : content) { hash ^= byte; hash *= 1099511628211ULL; }
    constexpr std::array names{"local", "type", "module", "link"};
    std::ostringstream spelling;
    // A double-underscore global identifier is implementation-reserved. It is
    // injected after preprocessing, so an illicit user macro cannot rewrite it.
    spelling << "__cellerator_generated_v1_" << names[static_cast<unsigned>(domain) - 1U]
             << '_' << std::hex << std::setw(16) << std::setfill('0') << hash;
    return {domain, hash, spelling.str(), true};
}

bool is_reserved_generated_identifier_v1(std::string_view spelling) noexcept {
    return spelling.rfind("__cellerator_generated_v1_", 0) == 0;
}

} // namespace Cellerator::compiler::frontend::source
