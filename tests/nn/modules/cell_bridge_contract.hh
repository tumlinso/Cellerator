#pragma once

// Include after either importing cellerator.nn or including Cellerator/nn.hh.
// These user-owned types deliberately stay outside the named module.
struct ProbeView { int value; };

struct ProbeInstance {
    int value;
    ProbeView read(cellerator::ViewTag<ProbeView>) const & { return {value}; }
};

struct ProbeModel {
    int operator()(ProbeInstance& instance, int increment) const {
        instance.value += increment;
        return instance.value;
    }
};

using ProbeCell = cellerator::Cell<ProbeModel, ProbeInstance>;

ProbeCell header_make_cell(ProbeModel& model, int value);
int header_cell_forward(ProbeCell& cell, int increment);
bool header_cell_view_matches(const ProbeCell& cell, int expected);
bool header_runtime_view_matches(const cellerator::CellViews& views, int expected);
