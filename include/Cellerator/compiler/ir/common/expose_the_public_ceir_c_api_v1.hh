#pragma once

#include <Cellerator/compiler/ir/common/implement_common_operation_and_extension_records_v1.hh>

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cellerator::compiler::ceir {

struct diagnostic { std::string message; std::size_t byte_begin{}; std::size_t byte_end{}; };
class context {
public:
    using extension_hook = std::function<bool(std::string_view)>;
    void register_extension(std::string namespace_name, extension_hook hook);
    bool accepts_extension(std::string_view qualified_name) const;
private:
    std::vector<std::pair<std::string, extension_hook>> extensions_;
};

class module {
public:
    using const_iterator = std::vector<ir::common_operation>::const_iterator;
    const_iterator begin() const noexcept;
    const_iterator end() const noexcept;
    std::size_t size() const noexcept;
private:
    friend class module_builder;
    explicit module(std::shared_ptr<const std::vector<ir::common_operation>> operations);
    std::shared_ptr<const std::vector<ir::common_operation>> operations_;
};

class module_builder {
public:
    explicit module_builder(context &owner) : owner_(&owner) {}
    bool append(ir::common_operation operation, diagnostic *error = nullptr);
    module freeze() const;
private:
    context *owner_;
    std::shared_ptr<std::vector<ir::common_operation>> operations_
        = std::make_shared<std::vector<ir::common_operation>>();
};

class writer {
public:
    std::string canonical(const module &value) const;
    std::string pretty(const module &value) const;
};

// Modules own immutable records. Iterators and views remain valid for the
// lifetime of the module value; builders retain independent mutable ownership.

} // namespace cellerator::compiler::ceir
