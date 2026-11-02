#pragma once

#include <Cellerator/cell.hh>
#include <typeinfo>

namespace cellerator {

template<class View> struct CellAccess;
template<class View> struct CellEdit;

// Optional runtime interface. A bridge borrows a Cell; its source must outlive
// it and remain at the same address. View lifetimes are governed separately.
class CellViews {
public:
    virtual ~CellViews() = 0;
    template<class View> View view() const &;
    template<class View> View view() const && = delete;
    template<class View> bool supports() const noexcept;
    template<class View> View edit() &;
    template<class View> View edit() && = delete;
    template<class View> bool supports_edit() const noexcept;
};

template<class View> struct CellAccess : virtual CellViews {
protected:
    friend class CellViews;
    virtual View read(ViewTag<View>) const & = 0;
};
template<class View> struct CellEdit : virtual CellViews {
protected:
    friend class CellViews;
    virtual View write(ViewTag<View>) & = 0;
};

template<class View> View CellViews::view() const & {
    return dynamic_cast<const CellAccess<View>&>(*this).read(ViewTag<View>{});
}
template<class View> bool CellViews::supports() const noexcept {
    return dynamic_cast<const CellAccess<View>*>(this) != nullptr;
}
template<class View> View CellViews::edit() & {
    return dynamic_cast<CellEdit<View>&>(*this).write(ViewTag<View>{});
}
template<class View> bool CellViews::supports_edit() const noexcept {
    return dynamic_cast<const CellEdit<View>*>(this) != nullptr;
}

namespace detail {
template<class Contract, class Source> class cell_view_contract;
template<class View, class Source>
class cell_view_contract<CellAccess<View>, Source> : public CellAccess<View> {
    static_assert(std::remove_const_t<Source>::template supports<View>(),
                  "selected read capability is unavailable");
public:
    explicit cell_view_contract(Source& source) : source_(&source) {}
protected:
    View read(ViewTag<View>) const & override { return source_->template view<View>(); }
private:
    Source* source_;
};
template<class View, class Source>
class cell_view_contract<CellEdit<View>, Source> : public CellEdit<View> {
    static_assert(!std::is_const_v<Source>, "const Cell cannot expose edit capabilities");
    static_assert(std::remove_const_t<Source>::template supports_edit<View>(),
                  "selected edit capability is unavailable");
public:
    explicit cell_view_contract(Source& source) : source_(&source) {}
protected:
    View write(ViewTag<View>) & override { return source_->template edit<View>(); }
private:
    Source* source_;
};

template<class Source, class... Contracts>
class cell_views_bridge final : public virtual CellViews,
                                public cell_view_contract<Contracts, Source>... {
public:
    explicit cell_views_bridge(Source& source)
        : cell_view_contract<Contracts, Source>(source)... {}
};
} // namespace detail

// A const Source& can bind a temporary, which would leave a dangling bridge.
template<class... Contracts, class Source,
         std::enable_if_t<!std::is_lvalue_reference_v<Source>, int> = 0>
void make_cell_views(Source&&) = delete;

template<class... Contracts, class Source>
auto make_cell_views(Source& source) {
    return detail::cell_views_bridge<Source, Contracts...>(source);
}

} // namespace cellerator
