#pragma once

#include <Cellerator/cell.hh>
#include <Cellerator/memory/view.hh>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cellerator::cells {

// Ordered token occurrences in the selected prefix. The pointer is borrowed;
// its placement is inherited from the source without any implicit transfer.
template<class Token>
struct EventView {
    memory::const_array_view<Token> tokens;
};

// An identified prefix into caller-owned bulk storage. The descriptor object
// must outlive this adapter. The owner maintains the data, its logical ordering,
// and synchronization. Updating the descriptor after growth permits an existing
// handle to reacquire the new pointer; already returned views may be invalid.
// Reordering requires constructing a new handle or explicitly calling rebind().
template<class Token, class Identity>
class EventPrefix final {
public:
    using buffer_type = memory::const_array_view<Token>;

    EventPrefix(const buffer_type& buffer, std::size_t offset, std::size_t limit,
                Identity identity)
        : buffer_(&buffer), offset_(offset), limit_(limit),
          identity_(std::move(identity)) {
        validate(buffer, offset, limit);
    }

    EventPrefix(const buffer_type&&, std::size_t, std::size_t, Identity) = delete;

    std::size_t offset() const noexcept { return offset_; }
    std::size_t limit() const noexcept { return limit_; }
    // Identifies the borrowed descriptor object, independent of the biological
    // identity or current data pointer. Its owner controls its lifetime.
    const buffer_type& buffer_descriptor() const & noexcept { return *buffer_; }
    const buffer_type& buffer_descriptor() const && = delete;
    const Identity& identity() const & noexcept { return identity_; }
    const Identity& identity() const && = delete;

    // Rebinding is caller-directed and validates the complete new selection
    // before changing this handle. The identity is unchanged.
    void rebind(const buffer_type& buffer, std::size_t offset, std::size_t limit) & {
        validate(buffer, offset, limit);
        buffer_ = &buffer;
        offset_ = offset;
        limit_ = limit;
    }

    void rebind(const buffer_type&&, std::size_t, std::size_t) & = delete;

    EventView<Token> read(ViewTag<EventView<Token>>) const & {
        const auto& buffer = *buffer_;
        validate(buffer, offset_, limit_);
        const Token* data = buffer.data;
        if (offset_ != 0) data += offset_;
        return {{data, limit_, buffer.where}};
    }
    EventView<Token> read(ViewTag<EventView<Token>>) const && = delete;

    CellIdentityView<Identity> read(ViewTag<CellIdentityView<Identity>>) const & {
        return {identity_};
    }
    CellIdentityView<Identity> read(ViewTag<CellIdentityView<Identity>>) const && = delete;

    // Standalone conveniences delegate directly to the public typed hooks.
    // Generic model bindings use these same hooks without runtime lookup.
    template<class View> static constexpr bool supports() noexcept {
        return cellerator::detail::can_read<EventPrefix, View>::value;
    }
    template<class View> static constexpr bool supports_edit() noexcept {
        return cellerator::detail::can_write<EventPrefix, View>::value;
    }
    template<class View, std::enable_if_t<
        cellerator::detail::can_read<EventPrefix, View>::value, int> = 0>
    View view() const & { return read(ViewTag<View>{}); }
    template<class View> void view() const && = delete;

private:
    static void validate(const buffer_type& buffer, std::size_t offset, std::size_t limit) {
        const auto max_bytes = std::numeric_limits<std::size_t>::max() / sizeof(Token);
        const auto max_difference = static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max());
        const auto max_elements = max_bytes < max_difference ? max_bytes : max_difference;
        if (buffer.count > max_elements)
            throw std::overflow_error("event buffer extent exceeds the addressable range");
        if (buffer.count != 0 && buffer.data == nullptr)
            throw std::invalid_argument("event buffer has a null data pointer");
        if (offset > buffer.count || limit > buffer.count - offset)
            throw std::out_of_range("event prefix is outside the buffer");
    }

    const buffer_type* buffer_;
    std::size_t offset_;
    std::size_t limit_;
    Identity identity_;
};

} // namespace cellerator::cells
