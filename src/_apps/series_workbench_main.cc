#include "series_workbench.hh"

#include <ncurses.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cctype>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace wb = ::cellerator::apps::workbench;

namespace {

enum class screen_id {
    sources = 0,
    datasets,
    parts,
    shards,
    output,
    run,
    inspect,
    preprocess,
};

enum class log_level {
    note = 0,
    success,
    warning,
    error
};

enum class action_id {
    none = 0,
    quit,
    switch_screen,
    next_screen,
    prev_screen,
    move_selection,
    page_selection,
    jump_start,
    jump_end,
    edit_active,
    cycle_source_format,
    toggle_source_include,
    prompt_manifest,
    prompt_series,
    run_conversion,
    run_preprocess,
    resize,
};

enum color_pair_id {
    cp_header = 1,
    cp_border,
    cp_title,
    cp_selection,
    cp_ok,
    cp_warn,
    cp_error,
    cp_footer,
    cp_meta,
    cp_heat_1,
    cp_heat_2,
    cp_heat_3,
    cp_heat_4,
    cp_heat_5,
    cp_heat_6,
};

struct action {
    action_id id = action_id::none;
    int value = 0;
};

struct log_entry {
    log_level level = log_level::note;
    std::string text;
};

struct rect {
    int y = 0;
    int x = 0;
    int h = 0;
    int w = 0;

    bool valid() const {
        return h > 0 && w > 0;
    }
};

struct ui_layout {
    int rows = 0;
    int cols = 0;
    bool too_small = false;
    bool compact = false;
    rect header;
    rect nav;
    rect main;
    rect summary;
    rect log;
    rect footer;
};

struct ui_windows {
    WINDOW *header = nullptr;
    WINDOW *nav = nullptr;
    WINDOW *main = nullptr;
    WINDOW *summary = nullptr;
    WINDOW *log = nullptr;
    WINDOW *footer = nullptr;
};

struct ui_runtime {
    ui_windows windows;
    bool curses_ready = false;
    bool colors_ready = false;
};

struct ui_state {
    screen_id active = screen_id::sources;
    wb::manifest_inspection inspection;
    wb::ingest_plan plan;
    wb::series_summary series;
    wb::conversion_report conversion;
    wb::preprocess_summary preprocess_run;
    wb::ingest_policy policy;
    wb::preprocess_config preprocess;
    std::vector<log_entry> log_lines;
    std::string last_manifest_path;
    std::string last_series_path;
    std::size_t selected_source = 0;
    std::size_t selected_dataset = 0;
    std::size_t selected_part = 0;
    std::size_t selected_shard = 0;
    int output_field = 0;
    int preprocess_field = 0;
    std::size_t source_scroll = 0;
    std::size_t dataset_scroll = 0;
    std::size_t part_scroll = 0;
    std::size_t shard_scroll = 0;
    std::size_t output_scroll = 0;
    std::size_t preprocess_scroll = 0;
    std::size_t inspect_scroll = 0;
};

struct ncurses_guard {
    ui_runtime *runtime = nullptr;

    ~ncurses_guard() {
        if (runtime == nullptr || !runtime->curses_ready) return;
        if (runtime->windows.header != nullptr) delwin(runtime->windows.header);
        if (runtime->windows.nav != nullptr) delwin(runtime->windows.nav);
        if (runtime->windows.main != nullptr) delwin(runtime->windows.main);
        if (runtime->windows.summary != nullptr) delwin(runtime->windows.summary);
        if (runtime->windows.log != nullptr) delwin(runtime->windows.log);
        if (runtime->windows.footer != nullptr) delwin(runtime->windows.footer);
        runtime->windows = {};
        curs_set(1);
        echo();
        nocbreak();
        keypad(stdscr, FALSE);
        endwin();
        runtime->curses_ready = false;
    }
};

constexpr int k_min_rows = 24;
constexpr int k_min_cols = 92;
constexpr std::size_t k_log_limit = 256u;
constexpr std::array<const char *, 8> screen_names = {
    "Sources", "Datasets", "Parts", "Shards", "Output", "Run", "Inspect", "Preprocess"
};
constexpr std::array<const char *, 5> output_labels = {
    "output_path", "max_part_nnz", "max_window_bytes", "reader_bytes", "verify_after_write"
};
constexpr std::array<const char *, 10> preprocess_labels = {
    "target_sum", "min_counts", "min_genes", "max_mito_fraction", "min_gene_sum",
    "min_detected", "min_variance", "device", "cache_dir", "mito_prefix"
};

int screen_index(screen_id screen) {
    return static_cast<int>(screen);
}

screen_id screen_from_index(int index) {
    const int count = static_cast<int>(screen_names.size());
    if (count <= 0) return screen_id::sources;
    index %= count;
    if (index < 0) index += count;
    return static_cast<screen_id>(index);
}

std::string trim_copy(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    return value;
}

std::string ellipsize(const std::string &text, int width) {
    if (width <= 0) return std::string();
    if (static_cast<int>(text.size()) <= width) return text;
    if (width <= 3) return text.substr(0, static_cast<std::size_t>(width));
    return text.substr(0, static_cast<std::size_t>(width - 3)) + "...";
}

std::string bool_name(bool value) {
    return value ? "true" : "false";
}

std::string compact_bytes(std::size_t bytes) {
    std::ostringstream oss;
    const double gib = static_cast<double>(bytes) / 1073741824.0;
    if (gib >= 1.0) {
        oss.setf(std::ios::fixed);
        oss.precision(2);
        oss << gib << " GiB";
        return oss.str();
    }
    const double mib = static_cast<double>(bytes) / 1048576.0;
    oss.setf(std::ios::fixed);
    oss.precision(1);
    oss << mib << " MiB";
    return oss.str();
}

log_level log_level_for_issue(wb::issue_severity severity) {
    switch (severity) {
        case wb::issue_severity::info: return log_level::note;
        case wb::issue_severity::warning: return log_level::warning;
        case wb::issue_severity::error: return log_level::error;
    }
    return log_level::note;
}

int color_attr(const ui_runtime &runtime, int pair_id) {
    return runtime.colors_ready ? COLOR_PAIR(pair_id) : 0;
}

int color_attr_for_log_level(const ui_runtime &runtime, log_level level) {
    if (!runtime.colors_ready) return 0;
    switch (level) {
        case log_level::success: return COLOR_PAIR(cp_ok);
        case log_level::warning: return COLOR_PAIR(cp_warn);
        case log_level::error: return COLOR_PAIR(cp_error);
        case log_level::note: break;
    }
    return 0;
}

void push_log(ui_state *ui, log_level level, const std::string &message) {
    if (ui == nullptr) return;
    ui->log_lines.push_back({level, message});
    if (ui->log_lines.size() > k_log_limit) {
        const std::size_t trim = std::min<std::size_t>(32u, ui->log_lines.size());
        ui->log_lines.erase(ui->log_lines.begin(), ui->log_lines.begin() + static_cast<std::ptrdiff_t>(trim));
    }
}

void push_issue_log(ui_state *ui, const wb::issue &entry) {
    if (ui == nullptr) return;
    push_log(ui,
             log_level_for_issue(entry.severity),
             wb::severity_name(entry.severity) + " " + entry.scope + ": " + entry.message);
}

std::size_t issue_count(const std::vector<wb::issue> &issues, wb::issue_severity severity) {
    std::size_t count = 0;
    for (const wb::issue &entry : issues) {
        if (entry.severity == severity) ++count;
    }
    return count;
}

void clamp_selection(ui_state *ui) {
    if (ui == nullptr) return;
    if (!ui->inspection.sources.empty()) ui->selected_source = std::min(ui->selected_source, ui->inspection.sources.size() - 1u);
    else ui->selected_source = 0;
    if (!ui->plan.datasets.empty()) ui->selected_dataset = std::min(ui->selected_dataset, ui->plan.datasets.size() - 1u);
    else ui->selected_dataset = 0;
    if (!ui->plan.parts.empty()) ui->selected_part = std::min(ui->selected_part, ui->plan.parts.size() - 1u);
    else ui->selected_part = 0;
    if (!ui->plan.shards.empty()) ui->selected_shard = std::min(ui->selected_shard, ui->plan.shards.size() - 1u);
    else ui->selected_shard = 0;
    ui->output_field = std::clamp(ui->output_field, 0, static_cast<int>(output_labels.size()) - 1);
    ui->preprocess_field = std::clamp(ui->preprocess_field, 0, static_cast<int>(preprocess_labels.size()) - 1);
}

void rebuild_plan(ui_state *ui) {
    if (ui == nullptr) return;
    ui->plan = wb::plan_series_ingest(ui->inspection.sources, ui->policy);
    clamp_selection(ui);
    std::ostringstream oss;
    oss << "planned " << ui->plan.datasets.size()
        << " dataset(s), " << ui->plan.parts.size()
        << " part(s), " << ui->plan.shards.size() << " shard(s)";
    push_log(ui, ui->plan.ok ? log_level::success : log_level::warning, oss.str());
    for (const wb::issue &entry : ui->plan.issues) push_issue_log(ui, entry);
}

void load_manifest(ui_state *ui, const std::string &path) {
    if (ui == nullptr) return;
    ui->last_manifest_path = path;
    ui->inspection = wb::inspect_manifest_tsv(path, ui->policy.reader_bytes);
    ui->conversion = {};
    ui->series = {};
    ui->preprocess_run = {};
    push_log(ui,
             ui->inspection.ok ? log_level::success : log_level::warning,
             (ui->inspection.ok ? "loaded manifest: " : "manifest has issues: ") + path);
    for (const wb::issue &entry : ui->inspection.issues) push_issue_log(ui, entry);
    rebuild_plan(ui);
}

void open_series(ui_state *ui, const std::string &path) {
    if (ui == nullptr) return;
    ui->last_series_path = path;
    ui->series = wb::summarize_series_csh5(path);
    push_log(ui,
             ui->series.ok ? log_level::success : log_level::error,
             (ui->series.ok ? "opened series: " : "failed to inspect series: ") + path);
    for (const wb::issue &entry : ui->series.issues) push_issue_log(ui, entry);
}

void run_conversion(ui_state *ui) {
    if (ui == nullptr) return;
    push_log(ui, log_level::note, "starting conversion");
    ui->conversion = wb::convert_plan_to_series_csh5(ui->plan);
    for (const wb::run_event &event : ui->conversion.events) {
        push_log(ui, log_level::note, event.phase + ": " + event.message);
    }
    for (const wb::issue &entry : ui->conversion.issues) push_issue_log(ui, entry);
    if (ui->conversion.ok) {
        open_series(ui, ui->policy.output_path);
        push_log(ui, log_level::success, "conversion completed");
    } else {
        push_log(ui, log_level::error, "conversion failed");
    }
}

void run_preprocess(ui_state *ui) {
    if (ui == nullptr) return;
    if (ui->series.path.empty()) {
        push_log(ui, log_level::warning, "no series.csh5 is open");
        return;
    }
    push_log(ui, log_level::note, "starting preprocess");
    ui->preprocess_run = wb::run_preprocess_pass(ui->series.path, ui->preprocess);
    for (const wb::issue &entry : ui->preprocess_run.issues) push_issue_log(ui, entry);
    if (ui->preprocess_run.ok) {
        std::ostringstream oss;
        oss << "preprocess kept_cells=" << ui->preprocess_run.kept_cells
            << " kept_genes=" << ui->preprocess_run.kept_genes;
        push_log(ui, log_level::success, oss.str());
    } else {
        push_log(ui, log_level::error, "preprocess failed");
    }
}

bool init_curses(ui_runtime *runtime) {
    if (runtime == nullptr) return false;
    if (initscr() == nullptr) return false;
    runtime->curses_ready = true;
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);
    set_escdelay(25);
    if (has_colors()) {
        start_color();
        use_default_colors();
        init_pair(cp_header, COLOR_WHITE, COLOR_BLUE);
        init_pair(cp_border, COLOR_CYAN, -1);
        init_pair(cp_title, COLOR_BLACK, COLOR_CYAN);
        init_pair(cp_selection, COLOR_BLACK, COLOR_YELLOW);
        init_pair(cp_ok, COLOR_GREEN, -1);
        init_pair(cp_warn, COLOR_YELLOW, -1);
        init_pair(cp_error, COLOR_RED, -1);
        init_pair(cp_footer, COLOR_BLACK, COLOR_CYAN);
        init_pair(cp_meta, COLOR_CYAN, -1);
        init_pair(cp_heat_1, COLOR_BLACK, COLOR_BLUE);
        init_pair(cp_heat_2, COLOR_BLACK, COLOR_CYAN);
        init_pair(cp_heat_3, COLOR_BLACK, COLOR_GREEN);
        init_pair(cp_heat_4, COLOR_BLACK, COLOR_YELLOW);
        init_pair(cp_heat_5, COLOR_WHITE, COLOR_RED);
        init_pair(cp_heat_6, COLOR_WHITE, COLOR_MAGENTA);
        runtime->colors_ready = true;
    }
    return true;
}

void destroy_window(WINDOW **win) {
    if (win == nullptr || *win == nullptr) return;
    delwin(*win);
    *win = nullptr;
}

bool sync_window(WINDOW **win, const rect &area) {
    if (!area.valid()) {
        destroy_window(win);
        return false;
    }
    if (*win == nullptr) {
        *win = newwin(area.h, area.w, area.y, area.x);
    } else {
        wresize(*win, area.h, area.w);
        mvwin(*win, area.y, area.x);
    }
    return *win != nullptr;
}

ui_layout compute_layout(int rows, int cols) {
    ui_layout layout;
    layout.rows = rows;
    layout.cols = cols;
    if (rows < k_min_rows || cols < k_min_cols) {
        layout.too_small = true;
        return layout;
    }

    constexpr int header_h = 3;
    constexpr int footer_h = 2;
    const int log_h = std::clamp(rows / 4, 7, 11);
    const int content_h = rows - header_h - footer_h - log_h;
    if (content_h < 12) {
        layout.too_small = true;
        return layout;
    }

    layout.header = {0, 0, header_h, cols};
    layout.log = {rows - footer_h - log_h, 0, log_h, cols};
    layout.footer = {rows - footer_h, 0, footer_h, cols};

    const bool wide = cols >= 122;
    layout.compact = !wide;
    if (wide) {
        const int nav_w = std::clamp(cols / 5, 24, 28);
        const int summary_w = std::clamp(cols / 4, 30, 36);
        const int main_w = cols - nav_w - summary_w;
        if (main_w < 40) {
            layout.too_small = true;
            return layout;
        }
        layout.nav = {header_h, 0, content_h, nav_w};
        layout.main = {header_h, nav_w, content_h, main_w};
        layout.summary = {header_h, nav_w + main_w, content_h, summary_w};
        return layout;
    }

    const int nav_w = 24;
    const int right_w = cols - nav_w;
    const int summary_h = 9;
    const int main_h = content_h - summary_h;
    if (right_w < 52 || main_h < 8) {
        layout.too_small = true;
        return layout;
    }
    layout.nav = {header_h, 0, content_h, nav_w};
    layout.main = {header_h, nav_w, main_h, right_w};
    layout.summary = {header_h + main_h, nav_w, summary_h, right_w};
    return layout;
}

void prepare_windows(ui_runtime *runtime, const ui_layout &layout) {
    if (runtime == nullptr) return;
    sync_window(&runtime->windows.header, layout.header);
    sync_window(&runtime->windows.nav, layout.nav);
    sync_window(&runtime->windows.main, layout.main);
    sync_window(&runtime->windows.summary, layout.summary);
    sync_window(&runtime->windows.log, layout.log);
    sync_window(&runtime->windows.footer, layout.footer);
}

void clear_inner_line(WINDOW *win, int y) {
    const int width = getmaxx(win) - 2;
    if (width <= 0 || y < 1 || y >= getmaxy(win) - 1) return;
    mvwhline(win, y, 1, ' ', width);
}

void draw_clipped(WINDOW *win, int y, int x, int max_width, const std::string &text, int attrs = 0) {
    if (win == nullptr || max_width <= 0) return;
    const std::string clipped = ellipsize(text, max_width);
    if (attrs != 0) wattron(win, attrs);
    mvwaddnstr(win, y, x, clipped.c_str(), max_width);
    if (attrs != 0) wattroff(win, attrs);
}

void draw_panel(WINDOW *win, const ui_runtime &runtime, const char *title) {
    if (win == nullptr) return;
    werase(win);
    if (runtime.colors_ready) wattron(win, COLOR_PAIR(cp_border));
    box(win, 0, 0);
    if (runtime.colors_ready) wattroff(win, COLOR_PAIR(cp_border));
    if (title != nullptr) {
        draw_clipped(win,
                     0,
                     2,
                     getmaxx(win) - 4,
                     std::string(" ") + title + " ",
                     color_attr(runtime, cp_title) | A_BOLD);
    }
}

int heatmap_attr(const ui_runtime &runtime, float value, float vmax) {
    if (!runtime.colors_ready || vmax <= 0.0f || value <= 0.0f) return 0;
    const float t = std::clamp(value / vmax, 0.0f, 1.0f);
    if (t < 0.16f) return COLOR_PAIR(cp_heat_1);
    if (t < 0.32f) return COLOR_PAIR(cp_heat_2);
    if (t < 0.48f) return COLOR_PAIR(cp_heat_3);
    if (t < 0.64f) return COLOR_PAIR(cp_heat_4);
    if (t < 0.82f) return COLOR_PAIR(cp_heat_5);
    return COLOR_PAIR(cp_heat_6);
}

int visible_rows_for_active(const ui_state &ui, const ui_layout &layout) {
    const int main_h = layout.main.h;
    if (main_h <= 0) return 1;
    switch (ui.active) {
        case screen_id::sources:
        case screen_id::datasets:
        case screen_id::parts:
        case screen_id::shards:
            return std::max(1, main_h - 6);
        case screen_id::output:
            return std::max(1, main_h - 6);
        case screen_id::preprocess:
            return std::max(1, main_h - 6);
        case screen_id::inspect:
            return std::max(1, main_h - 9);
        case screen_id::run:
            return std::max(1, main_h - 6);
    }
    return 1;
}

std::size_t *active_scroll(ui_state *ui) {
    if (ui == nullptr) return nullptr;
    switch (ui->active) {
        case screen_id::sources: return &ui->source_scroll;
        case screen_id::datasets: return &ui->dataset_scroll;
        case screen_id::parts: return &ui->part_scroll;
        case screen_id::shards: return &ui->shard_scroll;
        case screen_id::output: return &ui->output_scroll;
        case screen_id::preprocess: return &ui->preprocess_scroll;
        case screen_id::inspect: return &ui->inspect_scroll;
        case screen_id::run: break;
    }
    return nullptr;
}

std::size_t active_count(const ui_state &ui) {
    switch (ui.active) {
        case screen_id::sources: return ui.inspection.sources.size();
        case screen_id::datasets: return ui.plan.datasets.size();
        case screen_id::parts: return ui.plan.parts.size();
        case screen_id::shards: return ui.plan.shards.size();
        case screen_id::output: return output_labels.size();
        case screen_id::preprocess: return preprocess_labels.size();
        case screen_id::inspect: return ui.series.datasets.size();
        case screen_id::run: break;
    }
    return 0;
}

std::size_t active_selection(const ui_state &ui) {
    switch (ui.active) {
        case screen_id::sources: return ui.selected_source;
        case screen_id::datasets: return ui.selected_dataset;
        case screen_id::parts: return ui.selected_part;
        case screen_id::shards: return ui.selected_shard;
        case screen_id::output: return static_cast<std::size_t>(ui.output_field);
        case screen_id::preprocess: return static_cast<std::size_t>(ui.preprocess_field);
        case screen_id::inspect: return ui.inspect_scroll;
        case screen_id::run: break;
    }
    return 0;
}

void set_active_selection(ui_state *ui, std::size_t index) {
    if (ui == nullptr) return;
    switch (ui->active) {
        case screen_id::sources: ui->selected_source = index; break;
        case screen_id::datasets: ui->selected_dataset = index; break;
        case screen_id::parts: ui->selected_part = index; break;
        case screen_id::shards: ui->selected_shard = index; break;
        case screen_id::output: ui->output_field = static_cast<int>(index); break;
        case screen_id::preprocess: ui->preprocess_field = static_cast<int>(index); break;
        case screen_id::inspect: ui->inspect_scroll = index; break;
        case screen_id::run: break;
    }
}

void clamp_scroll(std::size_t *scroll, std::size_t selected, std::size_t total, int visible_rows) {
    if (scroll == nullptr) return;
    const std::size_t visible = static_cast<std::size_t>(std::max(1, visible_rows));
    if (total <= visible) {
        *scroll = 0;
        return;
    }
    const std::size_t max_scroll = total - visible;
    *scroll = std::min(*scroll, max_scroll);
    if (selected < *scroll) *scroll = selected;
    if (selected >= *scroll + visible) *scroll = selected + 1u - visible;
    *scroll = std::min(*scroll, max_scroll);
}

void clamp_viewports(ui_state *ui, const ui_layout &layout) {
    if (ui == nullptr || layout.too_small) return;
    clamp_selection(ui);
    const int visible = visible_rows_for_active(*ui, layout);
    clamp_scroll(active_scroll(ui), active_selection(*ui), active_count(*ui), visible);
}

std::string prompt_string(ui_runtime *runtime,
                          const ui_layout &layout,
                          const std::string &label,
                          const std::string &current = std::string()) {
    if (runtime == nullptr || runtime->windows.footer == nullptr || layout.too_small) return current;
    WINDOW *win = runtime->windows.footer;
    werase(win);
    if (runtime->colors_ready) wbkgd(win, COLOR_PAIR(cp_footer));
    const std::string prompt = current.empty() ? (label + ": ") : (label + " [" + current + "]: ");
    draw_clipped(win, 0, 1, getmaxx(win) - 2, prompt, A_BOLD | color_attr(*runtime, cp_footer));
    clear_inner_line(win, 1);
    wmove(win, 1, 1);
    wrefresh(win);
    echo();
    curs_set(1);
    char buffer[4096];
    buffer[0] = 0;
    wgetnstr(win, buffer, static_cast<int>(sizeof(buffer)) - 1);
    noecho();
    curs_set(0);
    if (runtime->colors_ready) wbkgd(win, 0);
    const std::string text = trim_copy(buffer);
    return text.empty() ? current : text;
}

unsigned long prompt_u64(ui_runtime *runtime, const ui_layout &layout, const std::string &label, unsigned long current) {
    const std::string text = prompt_string(runtime, layout, label, std::to_string(current));
    if (text.empty()) return current;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != 0) return current;
    return static_cast<unsigned long>(parsed);
}

float prompt_f32(ui_runtime *runtime, const ui_layout &layout, const std::string &label, float current) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << current;
    const std::string text = prompt_string(runtime, layout, label, oss.str());
    if (text.empty()) return current;
    char *end = nullptr;
    const float parsed = std::strtof(text.c_str(), &end);
    if (end == text.c_str() || *end != 0) return current;
    return parsed;
}

int prompt_i32(ui_runtime *runtime, const ui_layout &layout, const std::string &label, int current) {
    const std::string text = prompt_string(runtime, layout, label, std::to_string(current));
    if (text.empty()) return current;
    char *end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != 0) return current;
    return static_cast<int>(parsed);
}

void edit_active_field(ui_state *ui, ui_runtime *runtime, const ui_layout &layout) {
    if (ui == nullptr) return;
    switch (ui->active) {
        case screen_id::sources:
            if (!ui->inspection.sources.empty()) {
                wb::source_entry &source = ui->inspection.sources[ui->selected_source];
                source.dataset_id = prompt_string(runtime, layout, "dataset_id", source.dataset_id);
                rebuild_plan(ui);
            }
            break;
        case screen_id::output:
            switch (ui->output_field) {
                case 0: ui->policy.output_path = prompt_string(runtime, layout, "output_path", ui->policy.output_path); break;
                case 1: ui->policy.max_part_nnz = prompt_u64(runtime, layout, "max_part_nnz", ui->policy.max_part_nnz); break;
                case 2: ui->policy.max_window_bytes = prompt_u64(runtime, layout, "max_window_bytes", ui->policy.max_window_bytes); break;
                case 3: ui->policy.reader_bytes = static_cast<std::size_t>(prompt_u64(runtime, layout, "reader_bytes", static_cast<unsigned long>(ui->policy.reader_bytes))); break;
                case 4: ui->policy.verify_after_write = !ui->policy.verify_after_write; break;
            }
            rebuild_plan(ui);
            break;
        case screen_id::preprocess:
            switch (ui->preprocess_field) {
                case 0: ui->preprocess.target_sum = prompt_f32(runtime, layout, "target_sum", ui->preprocess.target_sum); break;
                case 1: ui->preprocess.min_counts = prompt_f32(runtime, layout, "min_counts", ui->preprocess.min_counts); break;
                case 2: ui->preprocess.min_genes = static_cast<unsigned int>(prompt_u64(runtime, layout, "min_genes", ui->preprocess.min_genes)); break;
                case 3: ui->preprocess.max_mito_fraction = prompt_f32(runtime, layout, "max_mito_fraction", ui->preprocess.max_mito_fraction); break;
                case 4: ui->preprocess.min_gene_sum = prompt_f32(runtime, layout, "min_gene_sum", ui->preprocess.min_gene_sum); break;
                case 5: ui->preprocess.min_detected_cells = prompt_f32(runtime, layout, "min_detected_cells", ui->preprocess.min_detected_cells); break;
                case 6: ui->preprocess.min_variance = prompt_f32(runtime, layout, "min_variance", ui->preprocess.min_variance); break;
                case 7: ui->preprocess.device = prompt_i32(runtime, layout, "device", ui->preprocess.device); break;
                case 8: ui->preprocess.cache_dir = prompt_string(runtime, layout, "cache_dir", ui->preprocess.cache_dir); break;
                case 9: ui->preprocess.mito_prefix = prompt_string(runtime, layout, "mito_prefix", ui->preprocess.mito_prefix); break;
            }
            break;
        case screen_id::datasets:
        case screen_id::parts:
        case screen_id::shards:
        case screen_id::run:
        case screen_id::inspect:
            break;
    }
}

void cycle_source_format(ui_state *ui) {
    if (ui == nullptr || ui->inspection.sources.empty()) return;
    wb::source_entry &source = ui->inspection.sources[ui->selected_source];
    switch (source.format) {
        case 0: source.format = cellerator::ingest::series::source_mtx; break;
        case cellerator::ingest::series::source_mtx: source.format = cellerator::ingest::series::source_tenx_mtx; break;
        default: source.format = cellerator::ingest::series::source_unknown; break;
    }
    rebuild_plan(ui);
}

void move_active_selection(ui_state *ui, int delta, int visible_rows) {
    if (ui == nullptr) return;
    const std::size_t total = active_count(*ui);
    if (total == 0) return;
    const int current = static_cast<int>(active_selection(*ui));
    const int next = std::clamp(current + delta, 0, static_cast<int>(total) - 1);
    set_active_selection(ui, static_cast<std::size_t>(next));
    clamp_scroll(active_scroll(ui), static_cast<std::size_t>(next), total, visible_rows);
}

void jump_active_selection(ui_state *ui, bool to_end, int visible_rows) {
    if (ui == nullptr) return;
    const std::size_t total = active_count(*ui);
    if (total == 0) return;
    const std::size_t next = to_end ? total - 1u : 0u;
    set_active_selection(ui, next);
    clamp_scroll(active_scroll(ui), next, total, visible_rows);
}

action map_input(int ch) {
    if (ch >= '1' && ch <= '8') {
        return {action_id::switch_screen, ch - '1'};
    }
    switch (ch) {
        case 'q': return {action_id::quit, 0};
        case '\t':
        case KEY_RIGHT: return {action_id::next_screen, 0};
        case KEY_BTAB:
        case KEY_LEFT: return {action_id::prev_screen, 0};
        case KEY_UP:
        case 'k': return {action_id::move_selection, -1};
        case KEY_DOWN:
        case 'j': return {action_id::move_selection, 1};
        case KEY_PPAGE: return {action_id::page_selection, -1};
        case KEY_NPAGE: return {action_id::page_selection, 1};
        case KEY_HOME:
        case 'g': return {action_id::jump_start, 0};
        case KEY_END:
        case 'G': return {action_id::jump_end, 0};
        case 'e':
        case '\n':
        case KEY_ENTER: return {action_id::edit_active, 0};
        case 'f': return {action_id::cycle_source_format, 0};
        case ' ': return {action_id::toggle_source_include, 0};
        case 'l': return {action_id::prompt_manifest, 0};
        case 'o': return {action_id::prompt_series, 0};
        case 'r': return {action_id::run_conversion, 0};
        case 'p': return {action_id::run_preprocess, 0};
        case KEY_RESIZE: return {action_id::resize, 0};
        default: break;
    }
    return {};
}

void fill_row(WINDOW *win, int y, int attrs = 0) {
    const int width = getmaxx(win) - 2;
    if (width <= 0 || y < 1 || y >= getmaxy(win) - 1) return;
    if (attrs != 0) wattron(win, attrs);
    mvwhline(win, y, 1, ' ', width);
    if (attrs != 0) wattroff(win, attrs);
}

void draw_list_row(WINDOW *win,
                   const ui_runtime &runtime,
                   int y,
                   bool selected,
                   const std::string &text,
                   int extra_attrs = 0) {
    const int width = getmaxx(win) - 4;
    const int attrs = (selected ? (color_attr(runtime, cp_selection) | A_BOLD) : 0) | extra_attrs;
    fill_row(win, y, attrs);
    draw_clipped(win, y, 2, width, text, attrs);
}

void draw_hint(WINDOW *win, const ui_runtime &runtime, int y, const std::string &text) {
    clear_inner_line(win, y);
    draw_clipped(win, y, 2, getmaxx(win) - 4, text, A_DIM | color_attr(runtime, cp_border));
}

void render_header(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    if (win == nullptr) return;
    werase(win);
    if (runtime.colors_ready) wbkgd(win, COLOR_PAIR(cp_header));
    for (int row = 0; row < getmaxy(win); ++row) mvwhline(win, row, 0, ' ', getmaxx(win));
    draw_clipped(win,
                 0,
                 2,
                 getmaxx(win) - 4,
                 "Cellerator Workbench  ingest workflow",
                 A_BOLD | color_attr(runtime, cp_header));

    std::ostringstream oss;
    oss << "Screen: " << screen_names[screen_index(ui.active)]
        << "  Plan: " << (ui.plan.ok ? "ready" : "issues")
        << "  Datasets: " << ui.plan.datasets.size()
        << "  Parts: " << ui.plan.parts.size()
        << "  Shards: " << ui.plan.shards.size();
    draw_clipped(win, 1, 2, getmaxx(win) - 4, oss.str(), color_attr(runtime, cp_header));

    const std::string path_line =
        "Manifest: " + (ui.last_manifest_path.empty() ? "<none>" : ui.last_manifest_path)
        + "  Output: " + (ui.policy.output_path.empty() ? "<unset>" : ui.policy.output_path);
    draw_clipped(win, 2, 2, getmaxx(win) - 4, path_line, color_attr(runtime, cp_header));
    if (runtime.colors_ready) wbkgd(win, 0);
}

void render_nav(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    if (win == nullptr) return;
    draw_panel(win, runtime, "Views");
    draw_hint(win, runtime, 1, "Tab/Shift-Tab or 1-8");
    for (int i = 0; i < static_cast<int>(screen_names.size()); ++i) {
        const bool selected = i == screen_index(ui.active);
        std::ostringstream oss;
        oss << (i + 1) << "  " << screen_names[static_cast<std::size_t>(i)];
        draw_list_row(win, runtime, 3 + i, selected, oss.str());
    }
    const int start = std::max(12, getmaxy(win) - 8);
    draw_hint(win, runtime, start, "l load manifest");
    draw_hint(win, runtime, start + 1, "o open series");
    draw_hint(win, runtime, start + 2, "r convert plan");
    draw_hint(win, runtime, start + 3, "p preprocess");
    draw_hint(win, runtime, start + 4, "e edit field");
    draw_hint(win, runtime, start + 5, "space toggle row");
}

void render_summary_status(WINDOW *win, const ui_runtime &runtime, int row, const std::string &label, bool ok) {
    clear_inner_line(win, row);
    draw_clipped(win, row, 2, getmaxx(win) - 4, label + ": ", A_BOLD);
    draw_clipped(win,
                 row,
                 2 + static_cast<int>(label.size()) + 2,
                 getmaxx(win) - 4 - static_cast<int>(label.size()) - 2,
                 ok ? "ok" : "needs attention",
                 (ok ? color_attr(runtime, cp_ok) : color_attr(runtime, cp_warn)) | A_BOLD);
}

void render_summary(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    if (win == nullptr) return;
    draw_panel(win, runtime, "Summary");
    render_summary_status(win, runtime, 1, "Manifest", ui.inspection.ok);
    render_summary_status(win, runtime, 2, "Plan", ui.plan.ok);
    render_summary_status(win, runtime, 3, "Series", ui.series.ok);
    render_summary_status(win, runtime, 4, "Preprocess", ui.preprocess_run.ok);

    std::ostringstream line5;
    line5 << "Rows " << ui.plan.total_rows << "  Cols " << ui.plan.total_cols;
    draw_clipped(win, 5, 2, getmaxx(win) - 4, line5.str());
    std::ostringstream line6;
    line6 << "NNZ " << ui.plan.total_nnz << "  Bytes " << compact_bytes(ui.plan.total_estimated_bytes);
    draw_clipped(win, 6, 2, getmaxx(win) - 4, line6.str());
    std::ostringstream line7;
    line7 << "Issues  err=" << issue_count(ui.plan.issues, wb::issue_severity::error)
          << " warn=" << issue_count(ui.plan.issues, wb::issue_severity::warning)
          << " info=" << issue_count(ui.plan.issues, wb::issue_severity::info);
    draw_clipped(win, 7, 2, getmaxx(win) - 4, line7.str());

    if (getmaxy(win) > 10) {
        const std::string series_line =
            ui.series.ok
                ? ("Series rows=" + std::to_string(ui.series.rows) + " parts=" + std::to_string(ui.series.num_parts))
                : "Series not open";
        draw_clipped(win, 9, 2, getmaxx(win) - 4, series_line);
    }
    if (getmaxy(win) > 11) {
        const std::string browse_line =
            ui.series.ok && ui.series.browse.available
                ? ("Browse cache ready: features=" + std::to_string(ui.series.browse.selected_feature_count))
                : "Browse cache unavailable";
        draw_clipped(win, 10, 2, getmaxx(win) - 4, browse_line);
    }
}

void render_sources(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    draw_panel(win, runtime, "Sources");
    draw_hint(win, runtime, 1, "Arrows/PgUp/PgDn move, e edit dataset id, f cycle format, space include");
    if (ui.inspection.sources.empty()) {
        draw_clipped(win, 3, 2, getmaxx(win) - 4, "Load a manifest with l to inspect MTX sources.", A_DIM);
        return;
    }
    const int visible = std::max(1, getmaxy(win) - 6);
    const std::size_t start = std::min(ui.source_scroll, ui.inspection.sources.size() - 1u);
    for (int row = 0; row < visible && start + static_cast<std::size_t>(row) < ui.inspection.sources.size(); ++row) {
        const std::size_t index = start + static_cast<std::size_t>(row);
        const wb::source_entry &source = ui.inspection.sources[index];
        std::ostringstream oss;
        oss << (source.included ? "[x] " : "[ ] ")
            << source.dataset_id
            << "  " << wb::format_name(source.format)
            << "  rows=" << source.rows
            << " cols=" << source.cols
            << " nnz=" << source.nnz
            << "  probe=" << (source.probe_ok ? "ok" : "pending");
        draw_list_row(win, runtime, 3 + row, index == ui.selected_source, oss.str());
    }
    const wb::source_entry &selected = ui.inspection.sources[ui.selected_source];
    const std::string detail = "matrix=" + selected.matrix_path + "  features=" + selected.feature_path;
    draw_hint(win, runtime, getmaxy(win) - 2, detail);
}

void render_datasets(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    draw_panel(win, runtime, "Datasets");
    draw_hint(win, runtime, 1, "Planned global row ranges and part counts");
    if (ui.plan.datasets.empty()) {
        draw_clipped(win, 3, 2, getmaxx(win) - 4, "No datasets planned yet.", A_DIM);
        return;
    }
    const int visible = std::max(1, getmaxy(win) - 6);
    for (int row = 0; row < visible && ui.dataset_scroll + static_cast<std::size_t>(row) < ui.plan.datasets.size(); ++row) {
        const std::size_t index = ui.dataset_scroll + static_cast<std::size_t>(row);
        const wb::planned_dataset &dataset = ui.plan.datasets[index];
        std::ostringstream oss;
        oss << dataset.dataset_id
            << "  rows=" << dataset.rows
            << " cols=" << dataset.cols
            << " nnz=" << dataset.nnz
            << " parts=" << dataset.part_count
            << "  row=[" << dataset.global_row_begin << "," << dataset.global_row_end << ")";
        draw_list_row(win, runtime, 3 + row, index == ui.selected_dataset, oss.str());
    }
    const wb::planned_dataset &selected = ui.plan.datasets[ui.selected_dataset];
    draw_hint(win, runtime, getmaxy(win) - 2, "feature_count=" + std::to_string(selected.feature_count) + "  barcode_count=" + std::to_string(selected.barcode_count));
}

void render_parts(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    draw_panel(win, runtime, "Parts");
    draw_hint(win, runtime, 1, "Per-part row windows and shard assignment");
    if (ui.plan.parts.empty()) {
        draw_clipped(win, 3, 2, getmaxx(win) - 4, "No parts planned yet.", A_DIM);
        return;
    }
    const int visible = std::max(1, getmaxy(win) - 6);
    for (int row = 0; row < visible && ui.part_scroll + static_cast<std::size_t>(row) < ui.plan.parts.size(); ++row) {
        const std::size_t index = ui.part_scroll + static_cast<std::size_t>(row);
        const wb::planned_part &part = ui.plan.parts[index];
        std::ostringstream oss;
        oss << "p" << part.part_id
            << "  " << part.dataset_id
            << "  rows=" << part.rows
            << " nnz=" << part.nnz
            << " shard=" << part.shard_id
            << "  row=[" << part.row_begin << "," << part.row_end << ")";
        draw_list_row(win, runtime, 3 + row, index == ui.selected_part, oss.str());
    }
    const wb::planned_part &selected = ui.plan.parts[ui.selected_part];
    draw_hint(win, runtime, getmaxy(win) - 2, "estimated_bytes=" + compact_bytes(selected.estimated_bytes));
}

void render_shards(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    draw_panel(win, runtime, "Shards");
    draw_hint(win, runtime, 1, "Shard packing over planned parts");
    if (ui.plan.shards.empty()) {
        draw_clipped(win, 3, 2, getmaxx(win) - 4, "No shards planned yet.", A_DIM);
        return;
    }
    const int visible = std::max(1, getmaxy(win) - 6);
    for (int row = 0; row < visible && ui.shard_scroll + static_cast<std::size_t>(row) < ui.plan.shards.size(); ++row) {
        const std::size_t index = ui.shard_scroll + static_cast<std::size_t>(row);
        const wb::planned_shard &shard = ui.plan.shards[index];
        std::ostringstream oss;
        oss << "s" << shard.shard_id
            << "  parts=[" << shard.part_begin << "," << shard.part_end << ")"
            << "  rows=" << shard.rows
            << " nnz=" << shard.nnz
            << "  bytes=" << compact_bytes(shard.estimated_bytes);
        draw_list_row(win, runtime, 3 + row, index == ui.selected_shard, oss.str());
    }
    const wb::planned_shard &selected = ui.plan.shards[ui.selected_shard];
    draw_hint(win, runtime, getmaxy(win) - 2, "row=[" + std::to_string(selected.row_begin) + "," + std::to_string(selected.row_end) + ")");
}

std::string output_value(const ui_state &ui, int index) {
    switch (index) {
        case 0: return ui.policy.output_path;
        case 1: return std::to_string(ui.policy.max_part_nnz);
        case 2: return std::to_string(ui.policy.max_window_bytes);
        case 3: return std::to_string(ui.policy.reader_bytes);
        case 4: return bool_name(ui.policy.verify_after_write);
        default: break;
    }
    return std::string();
}

void render_output(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    draw_panel(win, runtime, "Output");
    draw_hint(win, runtime, 1, "Edit output path and ingest sizing policy");
    const int visible = std::max(1, getmaxy(win) - 6);
    for (int row = 0; row < visible && ui.output_scroll + static_cast<std::size_t>(row) < output_labels.size(); ++row) {
        const int index = static_cast<int>(ui.output_scroll + static_cast<std::size_t>(row));
        std::ostringstream oss;
        oss << output_labels[static_cast<std::size_t>(index)] << "  " << output_value(ui, index);
        draw_list_row(win, runtime, 3 + row, index == ui.output_field, oss.str());
    }
    draw_hint(win, runtime, getmaxy(win) - 2, "Enter edits the selected field; booleans toggle immediately.");
}

void render_run(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    draw_panel(win, runtime, "Run");
    draw_hint(win, runtime, 1, "Press r to write the current plan to series.csh5");
    draw_clipped(win,
                 3,
                 2,
                 getmaxx(win) - 4,
                 std::string("Status: ") + (ui.conversion.ok ? "ok" : "not run / failed"),
                 (ui.conversion.ok ? color_attr(runtime, cp_ok) : color_attr(runtime, cp_warn)) | A_BOLD);
    const int visible = std::max(1, getmaxy(win) - 6);
    const int total = static_cast<int>(ui.conversion.events.size());
    const int start = std::max(0, total - visible);
    for (int row = 0; row < visible && start + row < total; ++row) {
        const wb::run_event &event = ui.conversion.events[static_cast<std::size_t>(start + row)];
        draw_clipped(win, 5 + row, 2, getmaxx(win) - 4, event.phase + ": " + event.message);
    }
}

void render_heatmap_preview(WINDOW *win,
                            const ui_runtime &runtime,
                            const wb::browse_cache_summary &browse,
                            int start_row,
                            int max_rows) {
    if (win == nullptr || !browse.available || browse.selected_feature_count == 0u || max_rows < 3) return;
    const std::size_t feature_count = std::min<std::size_t>(browse.selected_feature_names.size(), 8u);
    const std::size_t sample_rows = std::min<std::size_t>(browse.sample_rows_per_part, static_cast<std::size_t>(std::max(1, max_rows - 2)));
    if (feature_count == 0u || sample_rows == 0u || browse.part_sample_values.empty()) return;

    float vmax = 0.0f;
    for (std::size_t row = 0; row < sample_rows; ++row) {
        for (std::size_t feature = 0; feature < feature_count; ++feature) {
            const std::size_t index = row * browse.selected_feature_count + feature;
            if (index < browse.part_sample_values.size()) vmax = std::max(vmax, browse.part_sample_values[index]);
        }
    }

    draw_hint(win, runtime, start_row, "Sample heatmap for the first part");
    int row_y = start_row + 1;
    for (std::size_t row = 0; row < sample_rows && row_y < getmaxy(win) - 1; ++row, ++row_y) {
        std::ostringstream label;
        const std::uint64_t global_row =
            row < browse.part_sample_global_rows.size() ? browse.part_sample_global_rows[row] : std::numeric_limits<std::uint64_t>::max();
        if (global_row == std::numeric_limits<std::uint64_t>::max()) label << "row --";
        else label << "row " << global_row;
        draw_clipped(win, row_y, 2, 10, label.str(), color_attr(runtime, cp_meta));
        int x = 13;
        for (std::size_t feature = 0; feature < feature_count && x + 1 < getmaxx(win) - 1; ++feature) {
            const std::size_t index = row * browse.selected_feature_count + feature;
            const float value = index < browse.part_sample_values.size() ? browse.part_sample_values[index] : 0.0f;
            const int attrs = heatmap_attr(runtime, value, vmax);
            if (attrs != 0) wattron(win, attrs);
            mvwaddnstr(win, row_y, x, "  ", 2);
            if (attrs != 0) wattroff(win, attrs);
            x += 2;
        }
    }
    std::ostringstream legend;
    legend << "features: ";
    for (std::size_t feature = 0; feature < feature_count; ++feature) {
        if (feature != 0u) legend << ", ";
        legend << browse.selected_feature_names[feature];
    }
    draw_clipped(win, std::min(getmaxy(win) - 2, start_row + max_rows - 1), 2, getmaxx(win) - 4, legend.str(), A_DIM);
}

void render_inspect(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    draw_panel(win, runtime, "Inspect");
    if (!ui.series.ok) {
        draw_hint(win, runtime, 1, "Open a series.csh5 with o");
        draw_clipped(win, 3, 2, getmaxx(win) - 4, "The inspect view shows dataset and browse-cache metadata.", A_DIM);
        return;
    }
    draw_clipped(win, 1, 2, getmaxx(win) - 4, "path: " + ui.series.path);
    std::ostringstream line2;
    line2 << "rows=" << ui.series.rows << " cols=" << ui.series.cols << " nnz=" << ui.series.nnz
          << "  datasets=" << ui.series.num_datasets << " parts=" << ui.series.num_parts
          << " shards=" << ui.series.num_shards;
    draw_clipped(win, 2, 2, getmaxx(win) - 4, line2.str());
    const std::string browse_line =
        std::string("browse cache: ") + (ui.series.browse.available ? "available" : "missing")
        + "  top_features=" + std::to_string(ui.series.browse.selected_feature_count)
        + "  sample_rows_per_part=" + std::to_string(ui.series.browse.sample_rows_per_part);
    draw_clipped(win, 3, 2, getmaxx(win) - 4, browse_line);
    const std::string metadata_line =
        "embedded metadata tables=" + std::to_string(ui.series.embedded_metadata.size());
    draw_clipped(win, 4, 2, getmaxx(win) - 4, metadata_line, color_attr(runtime, cp_meta));
    draw_hint(win, runtime, 5, "Dataset table");
    const int visible = std::max(1, getmaxy(win) - 16);
    for (int row = 0; row < visible && ui.inspect_scroll + static_cast<std::size_t>(row) < ui.series.datasets.size(); ++row) {
        const wb::series_dataset_summary &dataset = ui.series.datasets[ui.inspect_scroll + static_cast<std::size_t>(row)];
        std::ostringstream oss;
        oss << dataset.dataset_id
            << "  rows=" << dataset.rows
            << " cols=" << dataset.cols
            << " nnz=" << dataset.nnz
            << "  format=" << wb::format_name(dataset.format);
        draw_clipped(win, 7 + row, 2, getmaxx(win) - 4, oss.str());
    }
    int row_base = 7 + visible + 1;
    if (!ui.series.embedded_metadata.empty() && row_base < getmaxy(win) - 5) {
        const wb::embedded_metadata_dataset_summary &metadata = ui.series.embedded_metadata.front();
        draw_hint(win, runtime, row_base, "Embedded metadata preview");
        std::ostringstream oss;
        oss << "dataset_index=" << metadata.dataset_index
            << " rows=" << metadata.rows
            << " cols=" << metadata.cols;
        draw_clipped(win, row_base + 1, 2, getmaxx(win) - 4, oss.str(), color_attr(runtime, cp_meta));
        std::ostringstream cols;
        cols << "columns: ";
        for (std::size_t i = 0; i < metadata.column_names.size() && i < 6u; ++i) {
            if (i != 0u) cols << ", ";
            cols << metadata.column_names[i];
        }
        draw_clipped(win, row_base + 2, 2, getmaxx(win) - 4, cols.str(), A_DIM);
        row_base += 4;
    }
    if (ui.series.browse.available && row_base < getmaxy(win) - 4) {
        render_heatmap_preview(win, runtime, ui.series.browse, row_base, getmaxy(win) - row_base - 2);
    }
    if (!ui.series.feature_names.empty()) {
        draw_hint(win, runtime, getmaxy(win) - 2, "first feature: " + ui.series.feature_names.front());
    }
}

std::string preprocess_value(const ui_state &ui, int index) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    switch (index) {
        case 0: oss << ui.preprocess.target_sum; return oss.str();
        case 1: oss << ui.preprocess.min_counts; return oss.str();
        case 2: return std::to_string(ui.preprocess.min_genes);
        case 3: oss << ui.preprocess.max_mito_fraction; return oss.str();
        case 4: oss << ui.preprocess.min_gene_sum; return oss.str();
        case 5: oss << ui.preprocess.min_detected_cells; return oss.str();
        case 6: oss << ui.preprocess.min_variance; return oss.str();
        case 7: return std::to_string(ui.preprocess.device);
        case 8: return ui.preprocess.cache_dir;
        case 9: return ui.preprocess.mito_prefix;
        default: break;
    }
    return std::string();
}

void render_preprocess(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    draw_panel(win, runtime, "Preprocess");
    draw_hint(win, runtime, 1, "Tune QC thresholds, then press p to run on the open series");
    const int visible = std::max(1, getmaxy(win) - 7);
    for (int row = 0; row < visible && ui.preprocess_scroll + static_cast<std::size_t>(row) < preprocess_labels.size(); ++row) {
        const int index = static_cast<int>(ui.preprocess_scroll + static_cast<std::size_t>(row));
        std::ostringstream oss;
        oss << preprocess_labels[static_cast<std::size_t>(index)] << "  " << preprocess_value(ui, index);
        draw_list_row(win, runtime, 3 + row, index == ui.preprocess_field, oss.str());
    }
    const std::string status = ui.preprocess_run.ok
                                   ? ("last run kept_cells=" + std::to_string(ui.preprocess_run.kept_cells)
                                      + " kept_genes=" + std::to_string(ui.preprocess_run.kept_genes))
                                   : "last run: not run / failed";
    draw_hint(win, runtime, getmaxy(win) - 2, status);
}

void render_main(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    if (win == nullptr) return;
    switch (ui.active) {
        case screen_id::sources: render_sources(win, runtime, ui); break;
        case screen_id::datasets: render_datasets(win, runtime, ui); break;
        case screen_id::parts: render_parts(win, runtime, ui); break;
        case screen_id::shards: render_shards(win, runtime, ui); break;
        case screen_id::output: render_output(win, runtime, ui); break;
        case screen_id::run: render_run(win, runtime, ui); break;
        case screen_id::inspect: render_inspect(win, runtime, ui); break;
        case screen_id::preprocess: render_preprocess(win, runtime, ui); break;
    }
}

void render_log(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    if (win == nullptr) return;
    draw_panel(win, runtime, "Log");
    const int visible = std::max(1, getmaxy(win) - 2);
    const int total = static_cast<int>(ui.log_lines.size());
    const int start = std::max(0, total - visible);
    for (int row = 0; row < visible && start + row < total; ++row) {
        const log_entry &entry = ui.log_lines[static_cast<std::size_t>(start + row)];
        draw_clipped(win,
                     1 + row,
                     2,
                     getmaxx(win) - 4,
                     entry.text,
                     color_attr_for_log_level(runtime, entry.level));
    }
}

void render_footer(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    if (win == nullptr) return;
    werase(win);
    if (runtime.colors_ready) wbkgd(win, COLOR_PAIR(cp_footer));
    for (int row = 0; row < getmaxy(win); ++row) mvwhline(win, row, 0, ' ', getmaxx(win));
    draw_clipped(win,
                 0,
                 1,
                 getmaxx(win) - 2,
                 "Keys: 1-8/TAB screens  arrows move  PgUp/PgDn page  g/G ends  e edit  l load  o open  r convert  p preprocess  q quit",
                 A_BOLD | color_attr(runtime, cp_footer));
    std::string hint;
    switch (ui.active) {
        case screen_id::sources: hint = "Sources focus persists selection and scroll state."; break;
        case screen_id::datasets: hint = "Datasets view shows global row placement."; break;
        case screen_id::parts: hint = "Parts view shows shard assignment and byte estimates."; break;
        case screen_id::shards: hint = "Shards view shows the packed output layout."; break;
        case screen_id::output: hint = "Output fields recompute the plan after edits."; break;
        case screen_id::run: hint = "Run uses the current plan and then opens the output series."; break;
        case screen_id::inspect: hint = "Inspect summarizes the written series and browse cache."; break;
        case screen_id::preprocess: hint = "Preprocess runs on the currently open series."; break;
    }
    draw_clipped(win, 1, 1, getmaxx(win) - 2, hint, color_attr(runtime, cp_footer));
    if (runtime.colors_ready) wbkgd(win, 0);
}

void render_too_small(const ui_runtime &runtime, const ui_layout &layout) {
    erase();
    const std::string title = "Cellerator Workbench";
    const std::string message = "Terminal too small. Resize to at least "
        + std::to_string(k_min_cols) + "x" + std::to_string(k_min_rows) + ".";
    const int center_y = std::max(0, layout.rows / 2 - 1);
    const int title_x = std::max(0, (layout.cols - static_cast<int>(title.size())) / 2);
    const int msg_x = std::max(0, (layout.cols - static_cast<int>(message.size())) / 2);
    if (runtime.colors_ready) attron(COLOR_PAIR(cp_title) | A_BOLD);
    mvaddnstr(center_y, title_x, title.c_str(), layout.cols);
    if (runtime.colors_ready) attroff(COLOR_PAIR(cp_title) | A_BOLD);
    mvaddnstr(center_y + 2, msg_x, message.c_str(), layout.cols);
    mvaddnstr(center_y + 4, std::max(0, msg_x - 8), "Press q to quit once resized if needed.", layout.cols);
    refresh();
}

void render_ui(ui_runtime *runtime, ui_state *ui, const ui_layout &layout) {
    if (runtime == nullptr || ui == nullptr) return;
    if (layout.too_small) {
        render_too_small(*runtime, layout);
        return;
    }
    clamp_viewports(ui, layout);
    prepare_windows(runtime, layout);
    render_header(runtime->windows.header, *runtime, *ui);
    render_nav(runtime->windows.nav, *runtime, *ui);
    render_main(runtime->windows.main, *runtime, *ui);
    render_summary(runtime->windows.summary, *runtime, *ui);
    render_log(runtime->windows.log, *runtime, *ui);
    render_footer(runtime->windows.footer, *runtime, *ui);
    wnoutrefresh(runtime->windows.header);
    wnoutrefresh(runtime->windows.nav);
    wnoutrefresh(runtime->windows.main);
    wnoutrefresh(runtime->windows.summary);
    wnoutrefresh(runtime->windows.log);
    wnoutrefresh(runtime->windows.footer);
    doupdate();
}

void handle_action(ui_state *ui, ui_runtime *runtime, const ui_layout &layout, const action &next, bool *running) {
    if (ui == nullptr || running == nullptr) return;
    switch (next.id) {
        case action_id::none:
            break;
        case action_id::quit:
            *running = false;
            break;
        case action_id::switch_screen:
            ui->active = screen_from_index(next.value);
            break;
        case action_id::next_screen:
            ui->active = screen_from_index(screen_index(ui->active) + 1);
            break;
        case action_id::prev_screen:
            ui->active = screen_from_index(screen_index(ui->active) - 1);
            break;
        case action_id::move_selection:
            move_active_selection(ui, next.value, visible_rows_for_active(*ui, layout));
            break;
        case action_id::page_selection:
            move_active_selection(ui, next.value * std::max(1, visible_rows_for_active(*ui, layout) - 1), visible_rows_for_active(*ui, layout));
            break;
        case action_id::jump_start:
            jump_active_selection(ui, false, visible_rows_for_active(*ui, layout));
            break;
        case action_id::jump_end:
            jump_active_selection(ui, true, visible_rows_for_active(*ui, layout));
            break;
        case action_id::edit_active:
            edit_active_field(ui, runtime, layout);
            break;
        case action_id::cycle_source_format:
            if (ui->active == screen_id::sources) cycle_source_format(ui);
            break;
        case action_id::toggle_source_include:
            if (ui->active == screen_id::sources && !ui->inspection.sources.empty()) {
                ui->inspection.sources[ui->selected_source].included = !ui->inspection.sources[ui->selected_source].included;
                rebuild_plan(ui);
            }
            break;
        case action_id::prompt_manifest: {
            const std::string path = prompt_string(runtime, layout, "manifest path", ui->last_manifest_path);
            if (!path.empty()) load_manifest(ui, path);
            break;
        }
        case action_id::prompt_series: {
            const std::string path = prompt_string(runtime, layout, "series path", ui->last_series_path);
            if (!path.empty()) open_series(ui, path);
            break;
        }
        case action_id::run_conversion:
            run_conversion(ui);
            break;
        case action_id::run_preprocess:
            run_preprocess(ui);
            break;
        case action_id::resize:
            resize_term(0, 0);
            clearok(stdscr, TRUE);
            break;
    }
}

void dump_plan(const wb::ingest_plan &plan) {
    std::printf("plan ok=%d datasets=%zu parts=%zu shards=%zu rows=%lu cols=%lu nnz=%lu bytes=%zu\n",
                plan.ok ? 1 : 0,
                plan.datasets.size(),
                plan.parts.size(),
                plan.shards.size(),
                plan.total_rows,
                plan.total_cols,
                plan.total_nnz,
                plan.total_estimated_bytes);
    for (const wb::planned_dataset &dataset : plan.datasets) {
        std::printf("dataset %s rows=%lu cols=%lu nnz=%lu parts=%lu row=[%lu,%lu)\n",
                    dataset.dataset_id.c_str(),
                    dataset.rows,
                    dataset.cols,
                    dataset.nnz,
                    dataset.part_count,
                    dataset.global_row_begin,
                    dataset.global_row_end);
    }
}

void dump_series(const wb::series_summary &summary) {
    std::printf("series ok=%d rows=%llu cols=%llu nnz=%llu datasets=%llu parts=%llu shards=%llu\n",
                summary.ok ? 1 : 0,
                static_cast<unsigned long long>(summary.rows),
                static_cast<unsigned long long>(summary.cols),
                static_cast<unsigned long long>(summary.nnz),
                static_cast<unsigned long long>(summary.num_datasets),
                static_cast<unsigned long long>(summary.num_parts),
                static_cast<unsigned long long>(summary.num_shards));
    for (const wb::series_dataset_summary &dataset : summary.datasets) {
        std::printf("dataset %s rows=%llu cols=%llu nnz=%llu format=%s\n",
                    dataset.dataset_id.c_str(),
                    static_cast<unsigned long long>(dataset.rows),
                    static_cast<unsigned long long>(dataset.cols),
                    static_cast<unsigned long long>(dataset.nnz),
                    wb::format_name(dataset.format).c_str());
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        ui_state ui;
        ui.policy.output_path = "series.csh5";
        push_log(&ui, log_level::note, "load a manifest with l or pass --manifest on the command line");
        push_log(&ui, log_level::note, "the workbench keeps per-screen scroll state for ingest planning");

        bool dump_plan_only = false;
        bool dump_series_only = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if ((arg == "--manifest" || arg == "-m") && i + 1 < argc) {
                ui.last_manifest_path = argv[++i];
            } else if ((arg == "--open" || arg == "-o") && i + 1 < argc) {
                ui.last_series_path = argv[++i];
            } else if (arg == "--dump-plan") {
                dump_plan_only = true;
            } else if (arg == "--dump-series") {
                dump_series_only = true;
            } else if ((arg == "--output" || arg == "-O") && i + 1 < argc) {
                ui.policy.output_path = argv[++i];
            } else {
                std::fprintf(stderr,
                             "Usage: %s [--manifest path.tsv] [--open path.csh5] [--output path.csh5] [--dump-plan] [--dump-series]\n",
                             argv[0]);
                return 2;
            }
        }

        if (!ui.last_manifest_path.empty()) load_manifest(&ui, ui.last_manifest_path);
        if (!ui.last_series_path.empty()) open_series(&ui, ui.last_series_path);

        if (dump_plan_only) {
            dump_plan(ui.plan);
            return ui.plan.ok ? 0 : 1;
        }
        if (dump_series_only) {
            dump_series(ui.series);
            return ui.series.ok ? 0 : 1;
        }

        ui_runtime runtime;
        if (!init_curses(&runtime)) {
            std::fprintf(stderr, "failed to initialize ncurses\n");
            return 1;
        }
        ncurses_guard guard{&runtime};

        bool running = true;
        while (running) {
            int rows = 0;
            int cols = 0;
            getmaxyx(stdscr, rows, cols);
            const ui_layout layout = compute_layout(rows, cols);
            render_ui(&runtime, &ui, layout);
            const int ch = getch();
            handle_action(&ui, &runtime, layout, map_input(ch), &running);
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "celleratorWorkbench: %s\n", ex.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "celleratorWorkbench: unknown error\n");
        return 1;
    }
}
