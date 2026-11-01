#pragma once

#include <typeinfo>

namespace cellerator {

template<class View> struct ViewTag {};
template<class View> struct CellAccess;
template<class View> struct CellEdit;

// A cell exposes only the access contracts implemented by its concrete type.
// The storage and lifetime of information exposed through a view are defined
// by that contract. Acquiring a view never implies ownership or a data copy.
class Cell {
public:
    virtual ~Cell() = 0;

    template<class View> View view() const &;
    template<class View> View view() const && = delete;
    template<class View> bool supports() const noexcept;

    template<class View> View edit() &;
    template<class View> View edit() && = delete;
    template<class View> bool supports_edit() const noexcept;
};

// read() is the protected implementation hook for a read-only contract.
// Cell::view() is the public entry point; a concrete Cell may implement any
// number of independent view types, including caller-defined types. Overrides
// should remain non-public so they cannot bypass the facade's rvalue guard.
template<class View>
struct CellAccess : virtual Cell {
protected:
    friend class Cell;
    virtual View read(ViewTag<View>) const & = 0;
};

// write() is the protected implementation hook for the separate edit
// capability. Cell::edit() is the public entry point. The owner of the
// returned borrowed view controls synchronization, generations, and
// invalidation; Cell does not infer or publish changes to backing data.
// Overrides should remain non-public and use the Cell::edit() facade.
template<class View>
struct CellEdit : virtual Cell {
protected:
    friend class Cell;
    virtual View write(ViewTag<View>) & = 0;
};

// A typed view of an application-defined biological identity. This reference
// is borrowed and cannot outlive the identity that the concrete Cell exposes.
template<class Identity>
struct CellIdentityView {
    const Identity& value;
};

template<class View>
View Cell::view() const & {
    return dynamic_cast<const CellAccess<View>&>(*this).read(ViewTag<View>{});
}

template<class View>
bool Cell::supports() const noexcept {
    return dynamic_cast<const CellAccess<View>*>(this) != nullptr;
}

template<class View>
View Cell::edit() & {
    return dynamic_cast<CellEdit<View>&>(*this).write(ViewTag<View>{});
}

template<class View>
bool Cell::supports_edit() const noexcept {
    return dynamic_cast<const CellEdit<View>*>(this) != nullptr;
}

} // namespace cellerator
