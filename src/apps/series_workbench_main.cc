#include "series_workbench.hh"

#include <ncurses.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cctype>
#include <exception>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace wb = ::cellerator::apps::workbench;

namespace {

enum class screen_id {
    home = 0,
    builder,
    sources,
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
    inspect_prev_focus,
    inspect_next_focus,
    inspect_cycle_mode,
    inspect_pan_columns,
    builder_toggle_mark,
    builder_parent_dir,
    builder_auto_group,
    builder_cycle_role,
    builder_assign,
    builder_new_dataset,
    builder_delete_dataset,
    builder_export_manifest,
    builder_apply_manifest,
    builder_clear_field,
    resize,
};

enum class inspect_focus_id {
    datasets = 0,
    metadata,
    heatmap,
};

enum class inspect_heatmap_mode {
    dataset_mean = 0,
    shard_mean,
    part_samples,
};

enum class builder_focus_id {
    browser = 0,
    drafts,
    detail,
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
    cp_home,
    cp_directory,
    cp_mark,
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

struct builder_state {
    std::string current_dir;
    std::vector<wb::filesystem_entry> entries;
    std::vector<std::string> marked_paths;
    std::vector<wb::draft_dataset> drafts;
    std::vector<wb::issue> issues;
    std::string export_path = "manifest.builder.tsv";
    std::size_t entry_selection = 0;
    std::size_t entry_scroll = 0;
    std::size_t draft_selection = 0;
    std::size_t draft_scroll = 0;
    std::size_t detail_selection = 0;
    std::size_t detail_scroll = 0;
    builder_focus_id focus = builder_focus_id::browser;
    wb::builder_path_role active_role = wb::builder_path_role::matrix;
};

struct ui_state {
    screen_id active = screen_id::home;
    wb::manifest_inspection inspection;
    wb::ingest_plan plan;
    wb::series_summary series;
    wb::conversion_report conversion;
    wb::preprocess_summary preprocess_run;
    wb::ingest_policy policy;
    wb::preprocess_config preprocess;
    builder_state builder;
    std::vector<log_entry> log_lines;
    std::string last_manifest_path;
    std::string last_series_path;
    int home_selection = 0;
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
    inspect_focus_id inspect_focus = inspect_focus_id::datasets;
    inspect_heatmap_mode inspect_mode = inspect_heatmap_mode::dataset_mean;
    std::size_t inspect_dataset_selection = 0;
    std::size_t inspect_dataset_scroll = 0;
    std::size_t inspect_metadata_selection = 0;
    std::size_t inspect_metadata_scroll = 0;
    std::size_t inspect_shard_selection = 0;
    std::size_t inspect_part_selection = 0;
    std::size_t inspect_heatmap_row_scroll = 0;
    std::size_t inspect_heatmap_col_scroll = 0;
    std::size_t inspect_metadata_table_index = std::numeric_limits<std::size_t>::max();
    wb::embedded_metadata_table inspect_metadata_table;
    struct metadata_value_count {
        std::string value;
        std::size_t count = 0;
    };
    std::size_t inspect_profile_column = std::numeric_limits<std::size_t>::max();
    std::size_t inspect_profile_unique = 0;
    std::vector<metadata_value_count> inspect_profile_values;
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
constexpr std::array<const char *, 10> screen_names = {
    "Home", "Builder", "Sources", "Datasets", "Parts", "Shards", "Output", "Run", "Inspect", "Preprocess"
};
constexpr std::array<const char *, 5> output_labels = {
    "output_path", "max_part_nnz", "max_window_bytes", "reader_bytes", "verify_after_write"
};
constexpr std::array<const char *, 10> preprocess_labels = {
    "target_sum", "min_counts", "min_genes", "max_mito_fraction", "min_gene_sum",
    "min_detected", "min_variance", "device", "cache_dir", "mito_prefix"
};
constexpr std::array<const char *, 7> builder_detail_labels = {
    "active_role", "export_path", "included", "matrix", "features", "barcodes", "metadata"
};

log_level log_level_for_issue(wb::issue_severity severity);
void push_log(ui_state *ui, log_level level, const std::string &message);
void push_issue_log(ui_state *ui, const wb::issue &entry);
void rebuild_plan(ui_state *ui);

int screen_index(screen_id screen) {
    return static_cast<int>(screen);
}

screen_id screen_from_index(int index) {
    const int count = static_cast<int>(screen_names.size());
    if (count <= 0) return screen_id::home;
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

const char *builder_focus_name(builder_focus_id focus) {
    switch (focus) {
        case builder_focus_id::browser: return "browser";
        case builder_focus_id::drafts: return "drafts";
        case builder_focus_id::detail: return "detail";
    }
    return "browser";
}

const char *home_action_name(int index) {
    switch (index) {
        case 0: return "Build Manifest";
        case 1: return "Load Manifest";
        case 2: return "Open Series";
    }
    return "Build Manifest";
}

std::size_t builder_mark_count(const builder_state &builder) {
    return builder.marked_paths.size();
}

bool builder_is_marked(const builder_state &builder, const std::string &path) {
    return std::find(builder.marked_paths.begin(), builder.marked_paths.end(), path) != builder.marked_paths.end();
}

void builder_remove_missing_marks(builder_state *builder) {
    if (builder == nullptr) return;
    std::vector<std::string> kept;
    kept.reserve(builder->marked_paths.size());
    for (const std::string &path : builder->marked_paths) {
        if (std::find_if(builder->entries.begin(),
                         builder->entries.end(),
                         [&](const wb::filesystem_entry &entry) { return entry.path == path; }) != builder->entries.end()) {
            kept.push_back(path);
        }
    }
    builder->marked_paths = std::move(kept);
}

void push_builder_issue_log(ui_state *ui, const wb::issue &entry) {
    if (ui == nullptr) return;
    push_log(ui, log_level_for_issue(entry.severity), "builder " + entry.scope + ": " + entry.message);
}

void refresh_builder_entries(ui_state *ui) {
    if (ui == nullptr) return;
    if (ui->builder.current_dir.empty()) {
        std::error_code ec;
        ui->builder.current_dir = std::filesystem::current_path(ec).string();
        if (ec || ui->builder.current_dir.empty()) ui->builder.current_dir = ".";
    }
    ui->builder.issues.clear();
    ui->builder.entries = wb::list_filesystem_entries(ui->builder.current_dir, &ui->builder.issues);
    builder_remove_missing_marks(&ui->builder);
    if (ui->builder.entries.empty()) ui->builder.entry_selection = 0;
    else ui->builder.entry_selection = std::min(ui->builder.entry_selection, ui->builder.entries.size() - 1u);
    for (const wb::issue &entry : ui->builder.issues) push_builder_issue_log(ui, entry);
}

void ensure_builder_draft(ui_state *ui) {
    if (ui == nullptr) return;
    if (!ui->builder.drafts.empty()) {
        ui->builder.draft_selection = std::min(ui->builder.draft_selection, ui->builder.drafts.size() - 1u);
        return;
    }
    wb::draft_dataset draft;
    draft.dataset_id = "dataset_1";
    ui->builder.drafts.push_back(std::move(draft));
    ui->builder.draft_selection = 0;
}

void builder_set_dir(ui_state *ui, const std::string &path) {
    if (ui == nullptr) return;
    ui->builder.current_dir = path;
    ui->builder.entry_selection = 0;
    ui->builder.entry_scroll = 0;
    ui->builder.marked_paths.clear();
    refresh_builder_entries(ui);
}

void auto_group_builder(ui_state *ui) {
    if (ui == nullptr) return;
    std::vector<wb::issue> issues;
    std::vector<wb::draft_dataset> drafts = wb::discover_dataset_drafts(ui->builder.current_dir, &issues);
    ui->builder.issues.insert(ui->builder.issues.end(), issues.begin(), issues.end());
    for (const wb::issue &entry : issues) push_builder_issue_log(ui, entry);
    if (!drafts.empty()) {
        ui->builder.drafts = std::move(drafts);
        ui->builder.draft_selection = 0;
        ui->builder.draft_scroll = 0;
        push_log(ui, log_level::success, "builder discovered manifest candidates in " + ui->builder.current_dir);
    } else {
        ensure_builder_draft(ui);
    }
}

const wb::filesystem_entry *builder_selected_entry(const ui_state &ui) {
    if (ui.builder.entry_selection >= ui.builder.entries.size()) return nullptr;
    return &ui.builder.entries[ui.builder.entry_selection];
}

wb::draft_dataset *builder_selected_draft(ui_state *ui) {
    if (ui == nullptr || ui->builder.draft_selection >= ui->builder.drafts.size()) return nullptr;
    return &ui->builder.drafts[ui->builder.draft_selection];
}

const wb::draft_dataset *builder_selected_draft(const ui_state &ui) {
    if (ui.builder.draft_selection >= ui.builder.drafts.size()) return nullptr;
    return &ui.builder.drafts[ui.builder.draft_selection];
}

std::string *builder_selected_detail_field(ui_state *ui) {
    wb::draft_dataset *draft = builder_selected_draft(ui);
    if (draft == nullptr) return nullptr;
    switch (ui->builder.detail_selection) {
        case 3: return &draft->matrix_path;
        case 4: return &draft->feature_path;
        case 5: return &draft->barcode_path;
        case 6: return &draft->metadata_path;
        default: return nullptr;
    }
}

void builder_toggle_mark(ui_state *ui) {
    if (ui == nullptr) return;
    const wb::filesystem_entry *entry = builder_selected_entry(*ui);
    if (entry == nullptr) return;
    const auto it = std::find(ui->builder.marked_paths.begin(), ui->builder.marked_paths.end(), entry->path);
    if (it != ui->builder.marked_paths.end()) {
        ui->builder.marked_paths.erase(it);
    } else {
        ui->builder.marked_paths.push_back(entry->path);
    }
}

std::vector<std::string> builder_assignment_paths(const ui_state &ui) {
    if (!ui.builder.marked_paths.empty()) return ui.builder.marked_paths;
    const wb::filesystem_entry *entry = builder_selected_entry(ui);
    if (entry == nullptr) return {};
    return {entry->path};
}

void assign_path_to_draft(wb::draft_dataset *draft, wb::builder_path_role role, const std::string &path) {
    if (draft == nullptr) return;
    switch (role) {
        case wb::builder_path_role::matrix: draft->matrix_path = path; break;
        case wb::builder_path_role::features: draft->feature_path = path; break;
        case wb::builder_path_role::barcodes: draft->barcode_path = path; break;
        case wb::builder_path_role::metadata: draft->metadata_path = path; break;
        case wb::builder_path_role::none: break;
    }
}

void builder_assign_selection(ui_state *ui) {
    if (ui == nullptr) return;
    ensure_builder_draft(ui);
    wb::draft_dataset *draft = builder_selected_draft(ui);
    if (draft == nullptr) return;
    const std::vector<std::string> paths = builder_assignment_paths(*ui);
    if (paths.empty()) return;
    std::size_t assigned = 0;
    for (const std::string &path : paths) {
        const auto it = std::find_if(ui->builder.entries.begin(),
                                     ui->builder.entries.end(),
                                     [&](const wb::filesystem_entry &entry) { return entry.path == path; });
        if (it == ui->builder.entries.end()) continue;
        if (!it->is_regular || !it->readable) {
            push_log(ui, log_level::warning, "builder skipped non-regular or unreadable path: " + path);
            continue;
        }
        wb::builder_path_role role = ui->builder.active_role;
        if (paths.size() > 1u) {
            const wb::builder_path_role inferred = wb::infer_builder_path_role(path);
            if (inferred != wb::builder_path_role::none) role = inferred;
        }
        assign_path_to_draft(draft, role, path);
        ++assigned;
    }
    if (draft->dataset_id.empty() && !draft->matrix_path.empty()) {
        const std::vector<wb::source_entry> inferred = wb::sources_from_dataset_drafts(std::vector<wb::draft_dataset>{*draft});
        if (!inferred.empty()) draft->dataset_id = inferred.front().dataset_id;
    }
    ui->builder.marked_paths.clear();
    if (assigned != 0u) {
        push_log(ui, log_level::success, "builder assigned " + std::to_string(assigned) + " path(s) to " + draft->dataset_id);
    }
}

void builder_apply_to_workbench(ui_state *ui) {
    if (ui == nullptr) return;
    ui->inspection = wb::inspect_source_entries(wb::sources_from_dataset_drafts(ui->builder.drafts),
                                                "<builder>",
                                                ui->policy.reader_bytes);
    ui->last_manifest_path = "<builder>";
    ui->conversion = {};
    for (const wb::issue &entry : ui->inspection.issues) push_issue_log(ui, entry);
    push_log(ui,
             ui->inspection.ok ? log_level::success : log_level::warning,
             ui->inspection.sources.empty() ? "builder produced no included datasets"
                                            : "builder generated an in-memory manifest");
    rebuild_plan(ui);
    ui->active = screen_id::sources;
}

void builder_export_manifest(ui_state *ui) {
    if (ui == nullptr) return;
    std::vector<wb::issue> issues;
    if (wb::export_manifest_tsv(ui->builder.export_path, ui->builder.drafts, ui->policy.reader_bytes, &issues)) {
        ui->last_manifest_path = ui->builder.export_path;
        push_log(ui, log_level::success, "exported manifest: " + ui->builder.export_path);
    } else {
        push_log(ui, log_level::error, "manifest export failed");
    }
    ui->builder.issues.insert(ui->builder.issues.end(), issues.begin(), issues.end());
    for (const wb::issue &entry : issues) push_builder_issue_log(ui, entry);
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

const char *inspect_focus_name(inspect_focus_id focus) {
    switch (focus) {
        case inspect_focus_id::datasets: return "datasets";
        case inspect_focus_id::metadata: return "metadata";
        case inspect_focus_id::heatmap: return "heatmap";
    }
    return "datasets";
}

const char *inspect_mode_name(inspect_heatmap_mode mode) {
    switch (mode) {
        case inspect_heatmap_mode::dataset_mean: return "dataset";
        case inspect_heatmap_mode::shard_mean: return "shards";
        case inspect_heatmap_mode::part_samples: return "part";
    }
    return "dataset";
}

bool ranges_overlap(std::uint64_t a_begin,
                    std::uint64_t a_end,
                    std::uint64_t b_begin,
                    std::uint64_t b_end) {
    return a_begin < b_end && b_begin < a_end;
}

void clamp_scroll(std::size_t *scroll, std::size_t selected, std::size_t total, int visible_rows);

void clear_inspect_profile(ui_state *ui) {
    if (ui == nullptr) return;
    ui->inspect_profile_column = std::numeric_limits<std::size_t>::max();
    ui->inspect_profile_unique = 0;
    ui->inspect_profile_values.clear();
}

void clear_inspect_metadata_cache(ui_state *ui) {
    if (ui == nullptr) return;
    ui->inspect_metadata_table_index = std::numeric_limits<std::size_t>::max();
    ui->inspect_metadata_table = {};
    ui->inspect_metadata_selection = 0;
    ui->inspect_metadata_scroll = 0;
    clear_inspect_profile(ui);
}

void reset_inspect_browser(ui_state *ui) {
    if (ui == nullptr) return;
    ui->inspect_focus = inspect_focus_id::datasets;
    ui->inspect_mode = inspect_heatmap_mode::dataset_mean;
    ui->inspect_dataset_selection = 0;
    ui->inspect_dataset_scroll = 0;
    ui->inspect_metadata_selection = 0;
    ui->inspect_metadata_scroll = 0;
    ui->inspect_shard_selection = 0;
    ui->inspect_part_selection = 0;
    ui->inspect_heatmap_row_scroll = 0;
    ui->inspect_heatmap_col_scroll = 0;
    clear_inspect_metadata_cache(ui);
}

const wb::series_dataset_summary *inspect_selected_dataset(const ui_state &ui) {
    if (ui.inspect_dataset_selection >= ui.series.datasets.size()) return nullptr;
    return &ui.series.datasets[ui.inspect_dataset_selection];
}

std::size_t inspect_metadata_table_for_dataset(const ui_state &ui, std::size_t dataset_index) {
    for (std::size_t i = 0; i < ui.series.embedded_metadata.size(); ++i) {
        if (ui.series.embedded_metadata[i].dataset_index == dataset_index) return i;
    }
    return std::numeric_limits<std::size_t>::max();
}

std::size_t inspect_part_count_for_dataset(const ui_state &ui, std::size_t dataset_index) {
    std::size_t count = 0;
    for (const wb::series_part_summary &part : ui.series.parts) {
        if (part.dataset_id == dataset_index) ++count;
    }
    return count;
}

std::size_t inspect_part_global_index(const ui_state &ui,
                                      std::size_t dataset_index,
                                      std::size_t local_index) {
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < ui.series.parts.size(); ++i) {
        if (ui.series.parts[i].dataset_id != dataset_index) continue;
        if (cursor == local_index) return i;
        ++cursor;
    }
    return std::numeric_limits<std::size_t>::max();
}

std::size_t inspect_shard_count_for_dataset(const ui_state &ui, std::size_t dataset_index) {
    if (dataset_index >= ui.series.datasets.size()) return 0;
    const wb::series_dataset_summary &dataset = ui.series.datasets[dataset_index];
    std::size_t count = 0;
    for (const wb::series_shard_summary &shard : ui.series.shards) {
        if (ranges_overlap(dataset.row_begin, dataset.row_end, shard.row_begin, shard.row_end)) ++count;
    }
    return count;
}

std::size_t inspect_shard_global_index(const ui_state &ui,
                                       std::size_t dataset_index,
                                       std::size_t local_index) {
    if (dataset_index >= ui.series.datasets.size()) return std::numeric_limits<std::size_t>::max();
    const wb::series_dataset_summary &dataset = ui.series.datasets[dataset_index];
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < ui.series.shards.size(); ++i) {
        if (!ranges_overlap(dataset.row_begin, dataset.row_end, ui.series.shards[i].row_begin, ui.series.shards[i].row_end)) continue;
        if (cursor == local_index) return i;
        ++cursor;
    }
    return std::numeric_limits<std::size_t>::max();
}

std::string inspect_metadata_value_at(const wb::embedded_metadata_table &table,
                                      std::uint64_t global_row,
                                      std::size_t column_index) {
    if (!table.available
        || global_row < table.row_begin
        || global_row >= table.row_end
        || column_index >= table.column_names.size()) return std::string();
    const std::size_t local_row = (std::size_t) (global_row - table.row_begin);
    if (local_row + 1u >= table.row_offsets.size()) return std::string();
    const std::size_t begin = table.row_offsets[local_row];
    const std::size_t end = table.row_offsets[local_row + 1u];
    const std::size_t field_index = begin + column_index;
    if (field_index >= end || field_index >= table.field_values.size()) return std::string();
    return table.field_values[field_index];
}

void rebuild_inspect_profile(ui_state *ui) {
    if (ui == nullptr) return;
    clear_inspect_profile(ui);
    const wb::embedded_metadata_table &table = ui->inspect_metadata_table;
    if (!table.available || table.rows == 0u || table.cols == 0u) return;
    if (ui->inspect_metadata_selection >= table.column_names.size()) return;

    std::unordered_map<std::string, std::size_t> counts;
    counts.reserve(std::min<std::size_t>(table.rows, 256u));
    for (std::size_t row = 0; row < table.rows; ++row) {
        if (row + 1u >= table.row_offsets.size()) break;
        const std::size_t begin = table.row_offsets[row];
        const std::size_t end = table.row_offsets[row + 1u];
        const std::size_t field_index = begin + ui->inspect_metadata_selection;
        if (field_index >= end || field_index >= table.field_values.size()) continue;
        ++counts[table.field_values[field_index]];
    }

    std::vector<ui_state::metadata_value_count> values;
    values.reserve(counts.size());
    for (const auto &entry : counts) values.push_back({entry.first, entry.second});
    std::sort(values.begin(),
              values.end(),
              [](const ui_state::metadata_value_count &lhs, const ui_state::metadata_value_count &rhs) {
                  if (lhs.count != rhs.count) return lhs.count > rhs.count;
                  return lhs.value < rhs.value;
              });
    ui->inspect_profile_column = ui->inspect_metadata_selection;
    ui->inspect_profile_unique = values.size();
    if (values.size() > 8u) values.resize(8u);
    ui->inspect_profile_values = std::move(values);
}

void ensure_inspect_metadata_loaded(ui_state *ui) {
    if (ui == nullptr || !ui->series.ok) return;
    if (ui->inspect_dataset_selection >= ui->series.datasets.size()) {
        clear_inspect_metadata_cache(ui);
        return;
    }

    const std::size_t table_index = inspect_metadata_table_for_dataset(*ui, ui->inspect_dataset_selection);
    if (table_index == std::numeric_limits<std::size_t>::max()) {
        clear_inspect_metadata_cache(ui);
        return;
    }
    if (ui->inspect_metadata_table_index == table_index && ui->inspect_metadata_table.available) {
        if (ui->inspect_profile_column != ui->inspect_metadata_selection) rebuild_inspect_profile(ui);
        return;
    }

    clear_inspect_metadata_cache(ui);
    ui->inspect_metadata_table_index = table_index;
    ui->inspect_metadata_table = wb::load_embedded_metadata_table(ui->series.path, table_index);
    if (ui->inspect_metadata_table.available && !ui->inspect_metadata_table.column_names.empty()) {
        ui->inspect_metadata_selection = std::min(ui->inspect_metadata_selection,
                                                  ui->inspect_metadata_table.column_names.size() - 1u);
        rebuild_inspect_profile(ui);
    } else {
        clear_inspect_profile(ui);
    }
}

void clamp_selection(ui_state *ui) {
    if (ui == nullptr) return;
    ui->home_selection = std::clamp(ui->home_selection, 0, 2);
    if (!ui->builder.entries.empty()) ui->builder.entry_selection = std::min(ui->builder.entry_selection, ui->builder.entries.size() - 1u);
    else ui->builder.entry_selection = 0;
    if (!ui->builder.drafts.empty()) ui->builder.draft_selection = std::min(ui->builder.draft_selection, ui->builder.drafts.size() - 1u);
    else ui->builder.draft_selection = 0;
    ui->builder.detail_selection = std::min<std::size_t>(ui->builder.detail_selection, builder_detail_labels.size() - 1u);
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

    if (!ui->series.datasets.empty()) ui->inspect_dataset_selection = std::min(ui->inspect_dataset_selection, ui->series.datasets.size() - 1u);
    else ui->inspect_dataset_selection = 0;
    ensure_inspect_metadata_loaded(ui);
    if (ui->inspect_metadata_table.available && !ui->inspect_metadata_table.column_names.empty()) {
        ui->inspect_metadata_selection = std::min(ui->inspect_metadata_selection,
                                                  ui->inspect_metadata_table.column_names.size() - 1u);
    } else {
        ui->inspect_metadata_selection = 0;
    }

    const std::size_t shard_count = inspect_shard_count_for_dataset(*ui, ui->inspect_dataset_selection);
    const std::size_t part_count = inspect_part_count_for_dataset(*ui, ui->inspect_dataset_selection);
    if (shard_count != 0u) ui->inspect_shard_selection = std::min(ui->inspect_shard_selection, shard_count - 1u);
    else ui->inspect_shard_selection = 0;
    if (part_count != 0u) ui->inspect_part_selection = std::min(ui->inspect_part_selection, part_count - 1u);
    else ui->inspect_part_selection = 0;
}

int inspect_dataset_visible_rows(const ui_layout &layout) {
    return std::max(1, layout.main.h - 10);
}

int inspect_metadata_visible_rows(const ui_layout &layout) {
    return std::max(1, layout.main.h - 10);
}

std::size_t inspect_visible_feature_count(const ui_layout &layout) {
    const int detail_width = layout.main.w - std::clamp(layout.main.w / 5, 22, 30) - std::clamp(layout.main.w / 4, 22, 28) - 8;
    const int label_width = 18;
    const int cells = std::max(1, (detail_width - label_width - 4) / 2);
    return (std::size_t) cells;
}

int inspect_heatmap_visible_rows(const ui_layout &layout) {
    return std::max(1, layout.main.h - 18);
}

int builder_browser_visible_rows(const ui_layout &layout) {
    return std::max(1, layout.main.h - 9);
}

int builder_draft_visible_rows(const ui_layout &layout) {
    return std::max(1, layout.main.h - 9);
}

int builder_detail_visible_rows(const ui_layout &layout) {
    return std::max(1, layout.main.h - 9);
}

void clamp_inspect_viewports(ui_state *ui, const ui_layout &layout) {
    if (ui == nullptr) return;
    clamp_scroll(&ui->inspect_dataset_scroll,
                 ui->inspect_dataset_selection,
                 ui->series.datasets.size(),
                 inspect_dataset_visible_rows(layout));
    clamp_scroll(&ui->inspect_metadata_scroll,
                 ui->inspect_metadata_selection,
                 ui->inspect_metadata_table.column_names.size(),
                 inspect_metadata_visible_rows(layout));

    const std::size_t feature_total = ui->series.browse.selected_feature_count;
    const std::size_t feature_visible = inspect_visible_feature_count(layout);
    if (feature_total <= feature_visible) ui->inspect_heatmap_col_scroll = 0;
    else ui->inspect_heatmap_col_scroll = std::min(ui->inspect_heatmap_col_scroll, feature_total - feature_visible);

    if (ui->inspect_mode == inspect_heatmap_mode::shard_mean) {
        const std::size_t shard_total = inspect_shard_count_for_dataset(*ui, ui->inspect_dataset_selection);
        if (shard_total == 0u) ui->inspect_heatmap_row_scroll = 0;
        else ui->inspect_heatmap_row_scroll = std::min(ui->inspect_heatmap_row_scroll, shard_total - 1u);
    } else if (ui->inspect_mode == inspect_heatmap_mode::part_samples) {
        const std::size_t local_part = ui->inspect_part_selection;
        const std::size_t part_index = inspect_part_global_index(*ui, ui->inspect_dataset_selection, local_part);
        std::size_t row_total = 0;
        if (part_index != std::numeric_limits<std::size_t>::max()
            && part_index + 1u < ui->series.browse.part_sample_row_offsets.size()) {
            row_total = (std::size_t) (ui->series.browse.part_sample_row_offsets[part_index + 1u]
                                       - ui->series.browse.part_sample_row_offsets[part_index]);
        }
        const std::size_t visible = (std::size_t) inspect_heatmap_visible_rows(layout);
        if (row_total <= visible) ui->inspect_heatmap_row_scroll = 0;
        else ui->inspect_heatmap_row_scroll = std::min(ui->inspect_heatmap_row_scroll, row_total - visible);
    } else {
        ui->inspect_heatmap_row_scroll = 0;
    }
}

void clamp_builder_viewports(ui_state *ui, const ui_layout &layout) {
    if (ui == nullptr) return;
    clamp_scroll(&ui->builder.entry_scroll,
                 ui->builder.entry_selection,
                 ui->builder.entries.size(),
                 builder_browser_visible_rows(layout));
    clamp_scroll(&ui->builder.draft_scroll,
                 ui->builder.draft_selection,
                 ui->builder.drafts.size(),
                 builder_draft_visible_rows(layout));
    clamp_scroll(&ui->builder.detail_scroll,
                 ui->builder.detail_selection,
                 builder_detail_labels.size(),
                 builder_detail_visible_rows(layout));
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
    reset_inspect_browser(ui);
    push_log(ui,
             ui->inspection.ok ? log_level::success : log_level::warning,
             (ui->inspection.ok ? "loaded manifest: " : "manifest has issues: ") + path);
    for (const wb::issue &entry : ui->inspection.issues) push_issue_log(ui, entry);
    rebuild_plan(ui);
    ui->active = screen_id::sources;
}

void open_series(ui_state *ui, const std::string &path) {
    if (ui == nullptr) return;
    ui->last_series_path = path;
    ui->series = wb::summarize_series_csh5(path);
    reset_inspect_browser(ui);
    push_log(ui,
             ui->series.ok ? log_level::success : log_level::error,
             (ui->series.ok ? "opened series: " : "failed to inspect series: ") + path);
    for (const wb::issue &entry : ui->series.issues) push_issue_log(ui, entry);
    if (ui->series.ok) ui->active = screen_id::inspect;
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
        init_pair(cp_header, COLOR_CYAN, -1);
        init_pair(cp_border, COLOR_CYAN, -1);
        init_pair(cp_title, COLOR_GREEN, -1);
        init_pair(cp_selection, COLOR_YELLOW, -1);
        init_pair(cp_ok, COLOR_GREEN, -1);
        init_pair(cp_warn, COLOR_YELLOW, -1);
        init_pair(cp_error, COLOR_RED, -1);
        init_pair(cp_footer, COLOR_YELLOW, -1);
        init_pair(cp_meta, COLOR_GREEN, -1);
        init_pair(cp_heat_1, COLOR_BLACK, COLOR_BLUE);
        init_pair(cp_heat_2, COLOR_BLACK, COLOR_CYAN);
        init_pair(cp_heat_3, COLOR_BLACK, COLOR_GREEN);
        init_pair(cp_heat_4, COLOR_BLACK, COLOR_YELLOW);
        init_pair(cp_heat_5, COLOR_WHITE, COLOR_RED);
        init_pair(cp_heat_6, COLOR_WHITE, COLOR_MAGENTA);
        init_pair(cp_home, COLOR_CYAN, -1);
        init_pair(cp_directory, COLOR_CYAN, -1);
        init_pair(cp_mark, COLOR_MAGENTA, -1);
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

    constexpr int header_h = 4;
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
        case screen_id::home:
            return 3;
        case screen_id::builder:
            switch (ui.builder.focus) {
                case builder_focus_id::browser: return builder_browser_visible_rows(layout);
                case builder_focus_id::drafts: return builder_draft_visible_rows(layout);
                case builder_focus_id::detail: return builder_detail_visible_rows(layout);
            }
            return std::max(1, main_h - 8);
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
            switch (ui.inspect_focus) {
                case inspect_focus_id::datasets: return inspect_dataset_visible_rows(layout);
                case inspect_focus_id::metadata: return inspect_metadata_visible_rows(layout);
                case inspect_focus_id::heatmap: return inspect_heatmap_visible_rows(layout);
            }
            return std::max(1, main_h - 9);
        case screen_id::run:
            return std::max(1, main_h - 6);
    }
    return 1;
}

std::size_t *active_scroll(ui_state *ui) {
    if (ui == nullptr) return nullptr;
    switch (ui->active) {
        case screen_id::home: break;
        case screen_id::builder:
            switch (ui->builder.focus) {
                case builder_focus_id::browser: return &ui->builder.entry_scroll;
                case builder_focus_id::drafts: return &ui->builder.draft_scroll;
                case builder_focus_id::detail: return &ui->builder.detail_scroll;
            }
            break;
        case screen_id::sources: return &ui->source_scroll;
        case screen_id::datasets: return &ui->dataset_scroll;
        case screen_id::parts: return &ui->part_scroll;
        case screen_id::shards: return &ui->shard_scroll;
        case screen_id::output: return &ui->output_scroll;
        case screen_id::preprocess: return &ui->preprocess_scroll;
        case screen_id::inspect:
            switch (ui->inspect_focus) {
                case inspect_focus_id::datasets: return &ui->inspect_dataset_scroll;
                case inspect_focus_id::metadata: return &ui->inspect_metadata_scroll;
                case inspect_focus_id::heatmap:
                    if (ui->inspect_mode == inspect_heatmap_mode::part_samples) return &ui->inspect_heatmap_row_scroll;
                    break;
            }
            break;
        case screen_id::run: break;
    }
    return nullptr;
}

std::size_t active_count(const ui_state &ui) {
    switch (ui.active) {
        case screen_id::home: return 3u;
        case screen_id::builder:
            switch (ui.builder.focus) {
                case builder_focus_id::browser: return ui.builder.entries.size();
                case builder_focus_id::drafts: return ui.builder.drafts.size();
                case builder_focus_id::detail: return builder_detail_labels.size();
            }
            break;
        case screen_id::sources: return ui.inspection.sources.size();
        case screen_id::datasets: return ui.plan.datasets.size();
        case screen_id::parts: return ui.plan.parts.size();
        case screen_id::shards: return ui.plan.shards.size();
        case screen_id::output: return output_labels.size();
        case screen_id::preprocess: return preprocess_labels.size();
        case screen_id::inspect:
            switch (ui.inspect_focus) {
                case inspect_focus_id::datasets: return ui.series.datasets.size();
                case inspect_focus_id::metadata: return ui.inspect_metadata_table.column_names.size();
                case inspect_focus_id::heatmap:
                    switch (ui.inspect_mode) {
                        case inspect_heatmap_mode::dataset_mean: return ui.series.datasets.empty() ? 0u : 1u;
                        case inspect_heatmap_mode::shard_mean: return inspect_shard_count_for_dataset(ui, ui.inspect_dataset_selection);
                        case inspect_heatmap_mode::part_samples: {
                            const std::size_t part_index =
                                inspect_part_global_index(ui, ui.inspect_dataset_selection, ui.inspect_part_selection);
                            if (part_index == std::numeric_limits<std::size_t>::max()
                                || part_index + 1u >= ui.series.browse.part_sample_row_offsets.size()) return 0u;
                            return (std::size_t) (ui.series.browse.part_sample_row_offsets[part_index + 1u]
                                                  - ui.series.browse.part_sample_row_offsets[part_index]);
                        }
                    }
                    break;
            }
            break;
        case screen_id::run: break;
    }
    return 0;
}

std::size_t active_selection(const ui_state &ui) {
    switch (ui.active) {
        case screen_id::home: return static_cast<std::size_t>(ui.home_selection);
        case screen_id::builder:
            switch (ui.builder.focus) {
                case builder_focus_id::browser: return ui.builder.entry_selection;
                case builder_focus_id::drafts: return ui.builder.draft_selection;
                case builder_focus_id::detail: return ui.builder.detail_selection;
            }
            break;
        case screen_id::sources: return ui.selected_source;
        case screen_id::datasets: return ui.selected_dataset;
        case screen_id::parts: return ui.selected_part;
        case screen_id::shards: return ui.selected_shard;
        case screen_id::output: return static_cast<std::size_t>(ui.output_field);
        case screen_id::preprocess: return static_cast<std::size_t>(ui.preprocess_field);
        case screen_id::inspect:
            switch (ui.inspect_focus) {
                case inspect_focus_id::datasets: return ui.inspect_dataset_selection;
                case inspect_focus_id::metadata: return ui.inspect_metadata_selection;
                case inspect_focus_id::heatmap:
                    switch (ui.inspect_mode) {
                        case inspect_heatmap_mode::dataset_mean: return 0u;
                        case inspect_heatmap_mode::shard_mean: return ui.inspect_shard_selection;
                        case inspect_heatmap_mode::part_samples: return ui.inspect_heatmap_row_scroll;
                    }
                    break;
            }
            break;
        case screen_id::run: break;
    }
    return 0;
}

void set_active_selection(ui_state *ui, std::size_t index) {
    if (ui == nullptr) return;
    switch (ui->active) {
        case screen_id::home:
            ui->home_selection = std::clamp<int>((int) index, 0, 2);
            break;
        case screen_id::builder:
            switch (ui->builder.focus) {
                case builder_focus_id::browser:
                    ui->builder.entry_selection = index;
                    break;
                case builder_focus_id::drafts:
                    ui->builder.draft_selection = index;
                    break;
                case builder_focus_id::detail:
                    ui->builder.detail_selection = index;
                    break;
            }
            break;
        case screen_id::sources: ui->selected_source = index; break;
        case screen_id::datasets: ui->selected_dataset = index; break;
        case screen_id::parts: ui->selected_part = index; break;
        case screen_id::shards: ui->selected_shard = index; break;
        case screen_id::output: ui->output_field = static_cast<int>(index); break;
        case screen_id::preprocess: ui->preprocess_field = static_cast<int>(index); break;
        case screen_id::inspect:
            switch (ui->inspect_focus) {
                case inspect_focus_id::datasets:
                    ui->inspect_dataset_selection = index;
                    ui->inspect_shard_selection = 0;
                    ui->inspect_part_selection = 0;
                    ui->inspect_heatmap_row_scroll = 0;
                    clear_inspect_metadata_cache(ui);
                    ensure_inspect_metadata_loaded(ui);
                    break;
                case inspect_focus_id::metadata:
                    ui->inspect_metadata_selection = index;
                    rebuild_inspect_profile(ui);
                    break;
                case inspect_focus_id::heatmap:
                    if (ui->inspect_mode == inspect_heatmap_mode::shard_mean) ui->inspect_shard_selection = index;
                    else if (ui->inspect_mode == inspect_heatmap_mode::part_samples) ui->inspect_heatmap_row_scroll = index;
                    break;
            }
            break;
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
    if (ui->active == screen_id::builder) {
        clamp_builder_viewports(ui, layout);
        return;
    }
    if (ui->active == screen_id::inspect) {
        clamp_inspect_viewports(ui, layout);
        return;
    }
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
        case screen_id::home:
            switch (ui->home_selection) {
                case 0:
                    ui->active = screen_id::builder;
                    break;
                case 1: {
                    const std::string path = prompt_string(runtime, layout, "manifest path", ui->last_manifest_path);
                    if (!path.empty()) load_manifest(ui, path);
                    break;
                }
                case 2: {
                    const std::string path = prompt_string(runtime, layout, "series path", ui->last_series_path);
                    if (!path.empty()) open_series(ui, path);
                    break;
                }
            }
            break;
        case screen_id::builder:
            ensure_builder_draft(ui);
            switch (ui->builder.focus) {
                case builder_focus_id::browser: {
                    const wb::filesystem_entry *entry = builder_selected_entry(*ui);
                    if (entry == nullptr) break;
                    if (entry->is_directory && entry->readable) {
                        builder_set_dir(ui, entry->path);
                    } else if (entry->is_regular && entry->readable) {
                        builder_assign_selection(ui);
                    }
                    break;
                }
                case builder_focus_id::drafts: {
                    wb::draft_dataset *draft = builder_selected_draft(ui);
                    if (draft != nullptr) draft->dataset_id = prompt_string(runtime, layout, "dataset_id", draft->dataset_id);
                    break;
                }
                case builder_focus_id::detail: {
                    wb::draft_dataset *draft = builder_selected_draft(ui);
                    if (draft == nullptr) break;
                    switch (ui->builder.detail_selection) {
                        case 0:
                            ui->builder.active_role =
                                ui->builder.active_role == wb::builder_path_role::metadata
                                    ? wb::builder_path_role::matrix
                                    : static_cast<wb::builder_path_role>(static_cast<int>(ui->builder.active_role) + 1);
                            break;
                        case 1:
                            ui->builder.export_path = prompt_string(runtime, layout, "manifest export", ui->builder.export_path);
                            break;
                        case 2:
                            draft->included = !draft->included;
                            break;
                        case 3:
                            draft->matrix_path.clear();
                            break;
                        case 4:
                            draft->feature_path.clear();
                            break;
                        case 5:
                            draft->barcode_path.clear();
                            break;
                        case 6:
                            draft->metadata_path.clear();
                            break;
                    }
                    break;
                }
            }
            break;
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
    if (ui->active == screen_id::inspect) {
        switch (ui->inspect_focus) {
            case inspect_focus_id::datasets: {
                const std::size_t total = ui->series.datasets.size();
                if (total == 0u) return;
                const int next = std::clamp((int) ui->inspect_dataset_selection + delta, 0, (int) total - 1);
                set_active_selection(ui, (std::size_t) next);
                clamp_scroll(&ui->inspect_dataset_scroll, (std::size_t) next, total, visible_rows);
                return;
            }
            case inspect_focus_id::metadata: {
                const std::size_t total = ui->inspect_metadata_table.column_names.size();
                if (total == 0u) return;
                const int next = std::clamp((int) ui->inspect_metadata_selection + delta, 0, (int) total - 1);
                set_active_selection(ui, (std::size_t) next);
                clamp_scroll(&ui->inspect_metadata_scroll, (std::size_t) next, total, visible_rows);
                return;
            }
            case inspect_focus_id::heatmap:
                if (ui->inspect_mode == inspect_heatmap_mode::shard_mean) {
                    const std::size_t total = inspect_shard_count_for_dataset(*ui, ui->inspect_dataset_selection);
                    if (total == 0u) return;
                    const int next = std::clamp((int) ui->inspect_shard_selection + delta, 0, (int) total - 1);
                    ui->inspect_shard_selection = (std::size_t) next;
                    ui->inspect_heatmap_row_scroll = ui->inspect_shard_selection;
                } else if (ui->inspect_mode == inspect_heatmap_mode::part_samples) {
                    const std::size_t total = inspect_part_count_for_dataset(*ui, ui->inspect_dataset_selection);
                    if (total == 0u) return;
                    const int next = std::clamp((int) ui->inspect_part_selection + delta, 0, (int) total - 1);
                    ui->inspect_part_selection = (std::size_t) next;
                    ui->inspect_heatmap_row_scroll = 0;
                }
                return;
        }
    }
    const std::size_t total = active_count(*ui);
    if (total == 0) return;
    const int current = static_cast<int>(active_selection(*ui));
    const int next = std::clamp(current + delta, 0, static_cast<int>(total) - 1);
    set_active_selection(ui, static_cast<std::size_t>(next));
    clamp_scroll(active_scroll(ui), static_cast<std::size_t>(next), total, visible_rows);
}

void jump_active_selection(ui_state *ui, bool to_end, int visible_rows) {
    if (ui == nullptr) return;
    if (ui->active == screen_id::inspect) {
        switch (ui->inspect_focus) {
            case inspect_focus_id::datasets: {
                const std::size_t total = ui->series.datasets.size();
                if (total == 0u) return;
                const std::size_t next = to_end ? total - 1u : 0u;
                set_active_selection(ui, next);
                clamp_scroll(&ui->inspect_dataset_scroll, next, total, visible_rows);
                return;
            }
            case inspect_focus_id::metadata: {
                const std::size_t total = ui->inspect_metadata_table.column_names.size();
                if (total == 0u) return;
                const std::size_t next = to_end ? total - 1u : 0u;
                set_active_selection(ui, next);
                clamp_scroll(&ui->inspect_metadata_scroll, next, total, visible_rows);
                return;
            }
            case inspect_focus_id::heatmap:
                if (ui->inspect_mode == inspect_heatmap_mode::shard_mean) {
                    const std::size_t total = inspect_shard_count_for_dataset(*ui, ui->inspect_dataset_selection);
                    if (total != 0u) {
                        ui->inspect_shard_selection = to_end ? total - 1u : 0u;
                        ui->inspect_heatmap_row_scroll = ui->inspect_shard_selection;
                    }
                } else if (ui->inspect_mode == inspect_heatmap_mode::part_samples) {
                    const std::size_t total = inspect_part_count_for_dataset(*ui, ui->inspect_dataset_selection);
                    if (total != 0u) ui->inspect_part_selection = to_end ? total - 1u : 0u;
                    ui->inspect_heatmap_row_scroll = 0;
                }
                return;
        }
    }
    const std::size_t total = active_count(*ui);
    if (total == 0) return;
    const std::size_t next = to_end ? total - 1u : 0u;
    set_active_selection(ui, next);
    clamp_scroll(active_scroll(ui), next, total, visible_rows);
}

action map_input(int ch) {
    if (ch >= '1' && ch <= '9') {
        return {action_id::switch_screen, ch - '1'};
    }
    if (ch == '0') return {action_id::switch_screen, 9};
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
        case 'l': return {action_id::prompt_manifest, 0};
        case 'o': return {action_id::prompt_series, 0};
        case 'r': return {action_id::run_conversion, 0};
        case 'p': return {action_id::run_preprocess, 0};
        case 'a': return {action_id::builder_assign, 0};
        case 'n': return {action_id::builder_new_dataset, 0};
        case 'u': return {action_id::builder_auto_group, 0};
        case 't': return {action_id::builder_cycle_role, 0};
        case 'x': return {action_id::builder_export_manifest, 0};
        case 'c': return {action_id::builder_apply_manifest, 0};
        case 'D': return {action_id::builder_delete_dataset, 0};
        case KEY_BACKSPACE:
        case 127: return {action_id::builder_parent_dir, 0};
        case '[': return {action_id::inspect_prev_focus, 0};
        case ']': return {action_id::inspect_next_focus, 0};
        case 'm': return {action_id::inspect_cycle_mode, 0};
        case ',': return {action_id::inspect_pan_columns, -1};
        case '.': return {action_id::inspect_pan_columns, 1};
        case 'z': return {action_id::builder_clear_field, 0};
        case ' ': return {action_id::builder_toggle_mark, 0};
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

void fill_span(WINDOW *win, int y, int x, int width, int attrs = 0) {
    if (win == nullptr || width <= 0 || y < 1 || y >= getmaxy(win) - 1) return;
    if (attrs != 0) wattron(win, attrs);
    mvwhline(win, y, x, ' ', width);
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

void draw_list_span_row(WINDOW *win,
                        const ui_runtime &runtime,
                        int y,
                        int x,
                        int width,
                        bool selected,
                        bool focused,
                        const std::string &text,
                        int extra_attrs = 0) {
    if (width <= 0) return;
    const int attrs = ((selected ? (focused ? (color_attr(runtime, cp_selection) | A_BOLD)
                                            : (color_attr(runtime, cp_meta) | A_BOLD))
                                 : 0)
                      | extra_attrs);
    fill_span(win, y, x, width, attrs);
    draw_clipped(win, y, x, width, text, attrs);
}

void draw_vertical_divider(WINDOW *win, const ui_runtime &runtime, int y, int x, int height) {
    if (win == nullptr || height <= 0 || x <= 0 || x >= getmaxx(win) - 1) return;
    if (runtime.colors_ready) wattron(win, COLOR_PAIR(cp_border));
    mvwvline(win, y, x, ACS_VLINE, height);
    if (runtime.colors_ready) wattroff(win, COLOR_PAIR(cp_border));
}

void draw_pane_title(WINDOW *win,
                     const ui_runtime &runtime,
                     int y,
                     int x,
                     int width,
                     bool focused,
                     const std::string &title) {
    const int attrs = (focused ? color_attr(runtime, cp_title) : color_attr(runtime, cp_border)) | A_BOLD;
    fill_span(win, y, x, width, focused ? color_attr(runtime, cp_title) : 0);
    draw_clipped(win, y, x, width, title, attrs);
}

void draw_hint(WINDOW *win, const ui_runtime &runtime, int y, const std::string &text) {
    clear_inner_line(win, y);
    draw_clipped(win, y, 2, getmaxx(win) - 4, text, A_DIM | color_attr(runtime, cp_border));
}

void render_header(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    if (win == nullptr) return;
    werase(win);
    if (runtime.colors_ready) wattron(win, COLOR_PAIR(cp_border));
    mvwhline(win, 0, 0, ACS_HLINE, getmaxx(win));
    mvwhline(win, getmaxy(win) - 1, 0, ACS_HLINE, getmaxx(win));
    if (runtime.colors_ready) wattroff(win, COLOR_PAIR(cp_border));
    draw_clipped(win,
                 1,
                 2,
                 getmaxx(win) - 4,
                 "Cellerator Workbench  bright ingest console",
                 A_BOLD | color_attr(runtime, cp_header));

    std::ostringstream oss;
    oss << "Screen: " << screen_names[screen_index(ui.active)];
    if (ui.active == screen_id::home) {
        oss << "  Start with Build Manifest, then review Sources and Run";
    } else if (ui.active == screen_id::builder) {
        oss << "  Focus: " << builder_focus_name(ui.builder.focus)
            << "  Role: " << wb::builder_path_role_name(ui.builder.active_role)
            << "  Drafts: " << ui.builder.drafts.size()
            << "  Marked: " << builder_mark_count(ui.builder);
    } else {
        oss << "  Plan: " << (ui.plan.ok ? "ready" : "issues")
            << "  Datasets: " << ui.plan.datasets.size()
            << "  Parts: " << ui.plan.parts.size()
            << "  Shards: " << ui.plan.shards.size();
    }
    draw_clipped(win, 2, 2, getmaxx(win) - 4, oss.str(), color_attr(runtime, cp_header));

    const std::string path_line = ui.active == screen_id::builder
                                      ? ("Directory: " + ui.builder.current_dir
                                         + "  Export: " + ui.builder.export_path)
                                      : ("Manifest: " + (ui.last_manifest_path.empty() ? "<none>" : ui.last_manifest_path)
                                         + "  Output: " + (ui.policy.output_path.empty() ? "<unset>" : ui.policy.output_path));
    draw_clipped(win, 3, 2, getmaxx(win) - 4, path_line, color_attr(runtime, cp_header));
}

void render_nav(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    if (win == nullptr) return;
    draw_panel(win, runtime, "Views");
    draw_hint(win, runtime, 1, "Tab/Shift-Tab or 1-9/0");
    for (int i = 0; i < static_cast<int>(screen_names.size()); ++i) {
        const bool selected = i == screen_index(ui.active);
        std::ostringstream oss;
        const char hotkey = i < 9 ? static_cast<char>('1' + i) : '0';
        oss << hotkey << "  " << screen_names[static_cast<std::size_t>(i)];
        draw_list_row(win, runtime, 3 + i, selected, oss.str());
    }
    const int start = std::max(3 + static_cast<int>(screen_names.size()) + 1, getmaxy(win) - 7);
    draw_hint(win, runtime, start, "e enter / act");
    draw_hint(win, runtime, start + 1, "l load manifest");
    draw_hint(win, runtime, start + 2, "o open series");
    draw_hint(win, runtime, start + 3, "r convert plan");
    draw_hint(win, runtime, start + 4, "p preprocess");
    draw_hint(win, runtime, start + 5, "space mark / toggle");
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
    if (ui.active == screen_id::home) {
        draw_panel(win, runtime, "Launch Summary");
        draw_clipped(win, 1, 2, getmaxx(win) - 4, "Manifest builder is now the primary entry path.", color_attr(runtime, cp_ok) | A_BOLD);
        draw_clipped(win, 3, 2, getmaxx(win) - 4, "Plan datasets: " + std::to_string(ui.plan.datasets.size()));
        draw_clipped(win, 4, 2, getmaxx(win) - 4, "Open series: " + (ui.series.ok ? ui.series.path : std::string("<none>")));
        draw_clipped(win, 6, 2, getmaxx(win) - 4, "Selected action: " + std::string(home_action_name(ui.home_selection)), color_attr(runtime, cp_home) | A_BOLD);
        draw_clipped(win, 8, 2, getmaxx(win) - 4, "Build Manifest creates sources directly in memory and can export TSV.", A_DIM);
        return;
    }
    if (ui.active == screen_id::builder) {
        draw_panel(win, runtime, "Builder Summary");
        draw_clipped(win, 1, 2, getmaxx(win) - 4, "Focus: " + std::string(builder_focus_name(ui.builder.focus)));
        draw_clipped(win, 2, 2, getmaxx(win) - 4, "Role: " + wb::builder_path_role_name(ui.builder.active_role), color_attr(runtime, cp_title) | A_BOLD);
        draw_clipped(win, 3, 2, getmaxx(win) - 4, "Drafts: " + std::to_string(ui.builder.drafts.size()) + "  Marked: " + std::to_string(builder_mark_count(ui.builder)));
        draw_clipped(win, 4, 2, getmaxx(win) - 4, "Export: " + ui.builder.export_path);
        draw_clipped(win, 6, 2, getmaxx(win) - 4, "Issues  err=" + std::to_string(issue_count(ui.builder.issues, wb::issue_severity::error))
                                                  + " warn=" + std::to_string(issue_count(ui.builder.issues, wb::issue_severity::warning))
                                                  + " info=" + std::to_string(issue_count(ui.builder.issues, wb::issue_severity::info)));
        const wb::draft_dataset *draft = builder_selected_draft(ui);
        if (draft != nullptr) {
            draw_clipped(win, 8, 2, getmaxx(win) - 4, "Selected draft: " + draft->dataset_id, color_attr(runtime, cp_ok) | A_BOLD);
            draw_clipped(win, 9, 2, getmaxx(win) - 4, draft->matrix_path.empty() ? "matrix missing" : draft->matrix_path, A_DIM);
        }
        return;
    }
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

void render_home(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    draw_panel(win, runtime, "Home");
    draw_clipped(win,
                 1,
                 2,
                 getmaxx(win) - 4,
                 "Design a manifest visually, validate it, then flow straight into ingest planning.",
                 color_attr(runtime, cp_home) | A_BOLD);
    draw_hint(win, runtime, 3, "Use arrows, then Enter. The builder can auto-group MTX triplets and export TSV.");

    const int card_w = getmaxx(win) - 6;
    for (int i = 0; i < 3; ++i) {
        const int y = 5 + i * 4;
        const bool selected = i == ui.home_selection;
        const int attrs = selected ? (color_attr(runtime, cp_selection) | A_BOLD) : (color_attr(runtime, cp_home) | A_BOLD);
        draw_clipped(win, y, 3, card_w, std::string(selected ? "> " : "  ") + home_action_name(i), attrs);
        if (i == 0) {
            draw_clipped(win, y + 1, 5, card_w - 2, "Browse directories like ncdu, assign roles, auto-build draft datasets.", A_DIM | color_attr(runtime, cp_home));
        } else if (i == 1) {
            draw_clipped(win, y + 1, 5, card_w - 2, "Inspect an existing manifest TSV and feed it into the planner.", A_DIM | color_attr(runtime, cp_home));
        } else {
            draw_clipped(win, y + 1, 5, card_w - 2, "Open a series.csh5 and jump directly into inspect and preprocess.", A_DIM | color_attr(runtime, cp_home));
        }
    }
    draw_hint(win, runtime, getmaxy(win) - 2, "Bright scientific palette, keyboard-first manifest design, existing heatmap inspector preserved.");
}

void render_builder(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    draw_panel(win, runtime, "Manifest Builder");
    draw_clipped(win, 1, 2, getmaxx(win) - 4, "dir: " + ui.builder.current_dir);
    draw_clipped(win,
                 2,
                 2,
                 getmaxx(win) - 4,
                 "focus=" + std::string(builder_focus_name(ui.builder.focus))
                     + "  role=" + wb::builder_path_role_name(ui.builder.active_role)
                     + "  marks=" + std::to_string(builder_mark_count(ui.builder))
                     + "  drafts=" + std::to_string(ui.builder.drafts.size()),
                 A_BOLD);
    draw_hint(win, runtime, 3, "Enter opens dirs or assigns files. Space marks. a assigns. u auto-groups. c applies. x exports.");

    const int inner_x = 2;
    const int inner_w = getmaxx(win) - 4;
    const int pane_top = 5;
    const int pane_h = getmaxy(win) - pane_top - 2;
    const int browser_w = std::clamp(inner_w / 2, 28, 38);
    const int drafts_w = std::clamp(inner_w / 4, 20, 28);
    const int detail_w = inner_w - browser_w - drafts_w - 4;
    if (pane_h < 8 || detail_w < 20) {
        draw_clipped(win, pane_top + 1, 2, getmaxx(win) - 4, "Builder needs a wider terminal for the three-pane layout.", A_DIM);
        return;
    }
    const rect browser{pane_top, inner_x, pane_h, browser_w};
    const rect drafts{pane_top, browser.x + browser.w + 2, pane_h, drafts_w};
    const rect detail{pane_top, drafts.x + drafts.w + 2, pane_h, detail_w};
    draw_vertical_divider(win, runtime, pane_top, browser.x + browser.w, pane_h);
    draw_vertical_divider(win, runtime, pane_top, drafts.x + drafts.w, pane_h);
    draw_pane_title(win, runtime, browser.y, browser.x, browser.w, ui.builder.focus == builder_focus_id::browser, "Browser");
    draw_pane_title(win, runtime, drafts.y, drafts.x, drafts.w, ui.builder.focus == builder_focus_id::drafts, "Drafts");
    draw_pane_title(win, runtime, detail.y, detail.x, detail.w, ui.builder.focus == builder_focus_id::detail, "Detail");

    const int browser_rows = browser.h - 3;
    for (int row = 0; row < browser_rows && ui.builder.entry_scroll + (std::size_t) row < ui.builder.entries.size(); ++row) {
        const std::size_t index = ui.builder.entry_scroll + (std::size_t) row;
        const wb::filesystem_entry &entry = ui.builder.entries[index];
        std::ostringstream oss;
        oss << (builder_is_marked(ui.builder, entry.path) ? "[*] " : "[ ] ")
            << (entry.is_directory ? "dir  " : (entry.is_regular ? "file " : "misc "))
            << entry.name;
        if (entry.is_regular) oss << "  " << compact_bytes((std::size_t) entry.size);
        const int extra = builder_is_marked(ui.builder, entry.path)
                              ? color_attr(runtime, cp_mark)
                              : (entry.is_directory ? color_attr(runtime, cp_directory) : 0);
        draw_list_span_row(win,
                           runtime,
                           browser.y + 2 + row,
                           browser.x,
                           browser.w,
                           index == ui.builder.entry_selection,
                           ui.builder.focus == builder_focus_id::browser,
                           oss.str(),
                           extra);
    }

    const int draft_rows = drafts.h - 3;
    for (int row = 0; row < draft_rows && ui.builder.draft_scroll + (std::size_t) row < ui.builder.drafts.size(); ++row) {
        const std::size_t index = ui.builder.draft_scroll + (std::size_t) row;
        const wb::draft_dataset &draft = ui.builder.drafts[index];
        std::ostringstream oss;
        oss << (draft.included ? "[x] " : "[ ] ")
            << draft.dataset_id
            << "  "
            << (draft.matrix_path.empty() ? "matrix?" : "matrix");
        draw_list_span_row(win,
                           runtime,
                           drafts.y + 2 + row,
                           drafts.x,
                           drafts.w,
                           index == ui.builder.draft_selection,
                           ui.builder.focus == builder_focus_id::drafts,
                           oss.str());
    }

    const wb::draft_dataset *draft = builder_selected_draft(ui);
    for (std::size_t i = 0; i < builder_detail_labels.size() && detail.y + 2 + (int) i < detail.y + detail.h - 1; ++i) {
        std::string value;
        if (i == 0) value = wb::builder_path_role_name(ui.builder.active_role);
        else if (i == 1) value = ui.builder.export_path;
        else if (i == 2) value = draft == nullptr ? "false" : bool_name(draft->included);
        else if (draft != nullptr && i == 3) value = draft->matrix_path;
        else if (draft != nullptr && i == 4) value = draft->feature_path;
        else if (draft != nullptr && i == 5) value = draft->barcode_path;
        else if (draft != nullptr && i == 6) value = draft->metadata_path;
        const std::string line = std::string(builder_detail_labels[i]) + "  " + (value.empty() ? "<unset>" : value);
        draw_list_span_row(win,
                           runtime,
                           detail.y + 2 + (int) i,
                           detail.x,
                           detail.w,
                           i == ui.builder.detail_selection,
                           ui.builder.focus == builder_focus_id::detail,
                           line,
                           i >= 3 ? color_attr(runtime, cp_meta) : 0);
    }
    draw_hint(win, runtime, detail.y + detail.h - 1, "z clears selected path field in Detail; D deletes selected draft.");
}

void render_sources(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    draw_panel(win, runtime, "Sources");
    draw_hint(win, runtime, 1, "Arrows/PgUp/PgDn move, e edit dataset id, f cycle format, space include");
    if (ui.inspection.sources.empty()) {
        draw_clipped(win, 3, 2, getmaxx(win) - 4, "Build or load a manifest to review source rows here.", A_DIM);
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

void render_inspect(WINDOW *win, const ui_runtime &runtime, const ui_state &ui) {
    draw_panel(win, runtime, "Inspect");
    if (!ui.series.ok) {
        draw_hint(win, runtime, 1, "Open a series.csh5 with o");
        draw_clipped(win, 3, 2, getmaxx(win) - 4, "The inspect view is the browser for metadata and precomputed heatmaps.", A_DIM);
        return;
    }

    draw_clipped(win, 1, 2, getmaxx(win) - 4, "path: " + ui.series.path);
    std::ostringstream line2;
    line2 << "rows=" << ui.series.rows << " cols=" << ui.series.cols << " nnz=" << ui.series.nnz
          << "  datasets=" << ui.series.num_datasets << " parts=" << ui.series.num_parts
          << " shards=" << ui.series.num_shards;
    draw_clipped(win, 2, 2, getmaxx(win) - 4, line2.str());
    std::ostringstream line3;
    line3 << "focus=" << inspect_focus_name(ui.inspect_focus)
          << "  mode=" << inspect_mode_name(ui.inspect_mode)
          << "  browse=" << (ui.series.browse.available ? "ready" : "missing")
          << "  metadata_tables=" << ui.series.embedded_metadata.size();
    draw_clipped(win, 3, 2, getmaxx(win) - 4, line3.str(), A_BOLD);
    draw_hint(win, runtime, 4, "[ and ] move focus, m switches dataset/shard/part heatmaps, , and . pan features");

    const int inner_x = 2;
    const int inner_w = getmaxx(win) - 4;
    const int pane_top = 6;
    const int pane_h = getmaxy(win) - pane_top - 2;
    int dataset_w = std::clamp(inner_w / 4, 20, 26);
    int metadata_w = std::clamp(inner_w / 4, 20, 28);
    int detail_w = inner_w - dataset_w - metadata_w - 4;
    if (detail_w < 20) {
        metadata_w = std::max(16, metadata_w - (20 - detail_w));
        detail_w = inner_w - dataset_w - metadata_w - 4;
    }
    if (detail_w < 16 || pane_h < 8) {
        draw_clipped(win, 7, 2, getmaxx(win) - 4, "Inspect view needs a wider terminal for the split heatmap browser.", A_DIM);
        return;
    }

    const rect dataset_pane{pane_top, inner_x, pane_h, dataset_w};
    const rect metadata_pane{pane_top, dataset_pane.x + dataset_pane.w + 2, pane_h, metadata_w};
    const rect detail_pane{pane_top, metadata_pane.x + metadata_pane.w + 2, pane_h, detail_w};

    draw_vertical_divider(win, runtime, pane_top, dataset_pane.x + dataset_pane.w, pane_h);
    draw_vertical_divider(win, runtime, pane_top, metadata_pane.x + metadata_pane.w, pane_h);

    draw_pane_title(win,
                    runtime,
                    dataset_pane.y,
                    dataset_pane.x,
                    dataset_pane.w,
                    ui.inspect_focus == inspect_focus_id::datasets,
                    "Datasets");
    draw_pane_title(win,
                    runtime,
                    metadata_pane.y,
                    metadata_pane.x,
                    metadata_pane.w,
                    ui.inspect_focus == inspect_focus_id::metadata,
                    "Metadata");
    draw_pane_title(win,
                    runtime,
                    detail_pane.y,
                    detail_pane.x,
                    detail_pane.w,
                    ui.inspect_focus == inspect_focus_id::heatmap,
                    "Detail / Heatmap");

    const int dataset_rows = dataset_pane.h - 3;
    for (int row = 0; row < dataset_rows
         && ui.inspect_dataset_scroll + (std::size_t) row < ui.series.datasets.size(); ++row) {
        const std::size_t index = ui.inspect_dataset_scroll + (std::size_t) row;
        const wb::series_dataset_summary &dataset = ui.series.datasets[index];
        std::ostringstream oss;
        oss << dataset.dataset_id << "  rows=" << dataset.rows << " nnz=" << dataset.nnz;
        draw_list_span_row(win,
                           runtime,
                           dataset_pane.y + 2 + row,
                           dataset_pane.x,
                           dataset_pane.w,
                           index == ui.inspect_dataset_selection,
                           ui.inspect_focus == inspect_focus_id::datasets,
                           oss.str());
    }

    const wb::series_dataset_summary *dataset = inspect_selected_dataset(ui);
    const bool have_metadata = ui.inspect_metadata_table.available && !ui.inspect_metadata_table.column_names.empty();
    if (!have_metadata) {
        const std::string message =
            ui.inspect_metadata_table.error.empty() ? "No embedded metadata for this dataset." : ui.inspect_metadata_table.error;
        draw_clipped(win,
                     metadata_pane.y + 2,
                     metadata_pane.x,
                     metadata_pane.w,
                     message,
                     A_DIM | color_attr(runtime, cp_meta));
    } else {
        const int metadata_rows = metadata_pane.h - 3;
        for (int row = 0; row < metadata_rows
             && ui.inspect_metadata_scroll + (std::size_t) row < ui.inspect_metadata_table.column_names.size(); ++row) {
            const std::size_t index = ui.inspect_metadata_scroll + (std::size_t) row;
            draw_list_span_row(win,
                               runtime,
                               metadata_pane.y + 2 + row,
                               metadata_pane.x,
                               metadata_pane.w,
                               index == ui.inspect_metadata_selection,
                               ui.inspect_focus == inspect_focus_id::metadata,
                               ui.inspect_metadata_table.column_names[index],
                               color_attr(runtime, cp_meta));
        }
    }

    int detail_row = detail_pane.y + 1;
    if (dataset != nullptr) {
        std::ostringstream dataset_line;
        dataset_line << dataset->dataset_id << "  format=" << wb::format_name(dataset->format)
                     << "  row=[" << dataset->row_begin << "," << dataset->row_end << ")";
        draw_clipped(win, detail_row++, detail_pane.x, detail_pane.w, dataset_line.str(), A_BOLD);
    }
    if (have_metadata && ui.inspect_metadata_selection < ui.inspect_metadata_table.column_names.size()) {
        std::ostringstream meta_line;
        meta_line << "column=" << ui.inspect_metadata_table.column_names[ui.inspect_metadata_selection]
                  << "  unique=" << ui.inspect_profile_unique
                  << "  rows=" << ui.inspect_metadata_table.rows;
        draw_clipped(win, detail_row++, detail_pane.x, detail_pane.w, meta_line.str(), color_attr(runtime, cp_meta));
        for (const ui_state::metadata_value_count &entry : ui.inspect_profile_values) {
            std::ostringstream value_line;
            value_line << entry.value << "  count=" << entry.count;
            draw_clipped(win, detail_row++, detail_pane.x, detail_pane.w, value_line.str(), A_DIM);
            if (detail_row >= detail_pane.y + 6) break;
        }
    } else {
        draw_clipped(win, detail_row++, detail_pane.x, detail_pane.w, "No metadata facet selected.", A_DIM);
    }

    if (!ui.series.browse.available || ui.series.browse.selected_feature_count == 0u) {
        draw_clipped(win, detail_row + 1, detail_pane.x, detail_pane.w, "Browse cache is missing, so no heatmap can be rendered.", A_DIM);
        return;
    }

    const std::size_t feature_total = ui.series.browse.selected_feature_count;
    const int label_width = std::min(20, std::max(14, detail_pane.w / 4));
    const int feature_cells = std::max(1, (detail_pane.w - label_width - 2) / 2);
    const std::size_t feature_begin = std::min(ui.inspect_heatmap_col_scroll, feature_total - 1u);
    const std::size_t feature_count = std::min<std::size_t>((std::size_t) feature_cells, feature_total - feature_begin);
    const int heatmap_top = detail_row + 1;
    const int heatmap_rows = std::max(1, detail_pane.y + detail_pane.h - heatmap_top - 2);
    const std::size_t stride = ui.series.browse.selected_feature_count;

    std::vector<std::string> row_labels;
    std::vector<const float *> row_ptrs;
    row_labels.reserve((std::size_t) heatmap_rows);
    row_ptrs.reserve((std::size_t) heatmap_rows);

    if (ui.inspect_mode == inspect_heatmap_mode::dataset_mean && dataset != nullptr) {
        const std::size_t offset = ui.inspect_dataset_selection * stride;
        if (offset + feature_total <= ui.series.browse.dataset_feature_mean.size()) {
            row_labels.push_back("dataset mean");
            row_ptrs.push_back(ui.series.browse.dataset_feature_mean.data() + offset);
        }
        draw_hint(win, runtime, detail_row, "Aggregate mean across selected browse features");
    } else if (ui.inspect_mode == inspect_heatmap_mode::shard_mean && dataset != nullptr) {
        const std::size_t shard_total = inspect_shard_count_for_dataset(ui, ui.inspect_dataset_selection);
        const std::size_t shard_begin = std::min(ui.inspect_heatmap_row_scroll, shard_total == 0u ? 0u : shard_total - 1u);
        for (std::size_t local = shard_begin;
             local < shard_total && row_labels.size() < (std::size_t) heatmap_rows;
             ++local) {
            const std::size_t global_index = inspect_shard_global_index(ui, ui.inspect_dataset_selection, local);
            if (global_index == std::numeric_limits<std::size_t>::max()) continue;
            const std::size_t offset = global_index * stride;
            if (offset + feature_total > ui.series.browse.shard_feature_mean.size()) continue;
            std::ostringstream label;
            label << (local == ui.inspect_shard_selection ? "> " : "  ")
                  << "s" << ui.series.shards[global_index].shard_id;
            row_labels.push_back(label.str());
            row_ptrs.push_back(ui.series.browse.shard_feature_mean.data() + offset);
        }
        std::ostringstream mode_line;
        mode_line << "Shard means  selected=" << (ui.inspect_shard_selection + 1u)
                  << "/" << std::max<std::size_t>(1u, shard_total);
        draw_hint(win, runtime, detail_row, mode_line.str());
    } else if (ui.inspect_mode == inspect_heatmap_mode::part_samples && dataset != nullptr) {
        const std::size_t part_total = inspect_part_count_for_dataset(ui, ui.inspect_dataset_selection);
        const std::size_t part_index = inspect_part_global_index(ui, ui.inspect_dataset_selection, ui.inspect_part_selection);
        if (part_index != std::numeric_limits<std::size_t>::max()
            && part_index + 1u < ui.series.browse.part_sample_row_offsets.size()) {
            const std::size_t row_begin = ui.series.browse.part_sample_row_offsets[part_index];
            const std::size_t row_end = ui.series.browse.part_sample_row_offsets[part_index + 1u];
            std::ostringstream mode_line;
            mode_line << "Part samples  selected=" << (ui.inspect_part_selection + 1u)
                      << "/" << std::max<std::size_t>(1u, part_total)
                      << "  p" << ui.series.parts[part_index].part_id
                      << " row=[" << ui.series.parts[part_index].row_begin
                      << "," << ui.series.parts[part_index].row_end << ")";
            draw_hint(win, runtime, detail_row, mode_line.str());
            for (std::size_t row = row_begin + ui.inspect_heatmap_row_scroll;
                 row < row_end && row_labels.size() < (std::size_t) heatmap_rows;
                 ++row) {
                const std::uint64_t global_row =
                    row < ui.series.browse.part_sample_global_rows.size()
                        ? ui.series.browse.part_sample_global_rows[row]
                        : std::numeric_limits<std::uint64_t>::max();
                std::ostringstream label;
                label << "row ";
                if (global_row == std::numeric_limits<std::uint64_t>::max()) label << "--";
                else label << global_row;
                const std::string metadata_value =
                    inspect_metadata_value_at(ui.inspect_metadata_table, global_row, ui.inspect_metadata_selection);
                if (!metadata_value.empty()) label << " " << metadata_value;
                row_labels.push_back(label.str());
                const std::size_t value_offset = row * stride;
                if (value_offset + feature_total <= ui.series.browse.part_sample_values.size()) {
                    row_ptrs.push_back(ui.series.browse.part_sample_values.data() + value_offset);
                } else {
                    row_ptrs.push_back(nullptr);
                }
            }
        } else {
            draw_hint(win, runtime, detail_row, "No sampled rows are available for the selected part.");
        }
    }

    float vmax = 0.0f;
    for (const float *row_ptr : row_ptrs) {
        if (row_ptr == nullptr) continue;
        for (std::size_t feature = 0; feature < feature_count; ++feature) {
            vmax = std::max(vmax, row_ptr[feature_begin + feature]);
        }
    }
    if (row_ptrs.empty()) {
        draw_clipped(win, heatmap_top, detail_pane.x, detail_pane.w, "No heatmap values available for the current selection.", A_DIM);
        return;
    }

    for (std::size_t row = 0; row < row_ptrs.size() && heatmap_top + (int) row < detail_pane.y + detail_pane.h - 1; ++row) {
        const int y = heatmap_top + (int) row;
        draw_clipped(win, y, detail_pane.x, label_width, row_labels[row], color_attr(runtime, cp_meta));
        int x = detail_pane.x + label_width + 1;
        for (std::size_t feature = 0; feature < feature_count && x + 1 < detail_pane.x + detail_pane.w; ++feature) {
            const float value = row_ptrs[row] != nullptr ? row_ptrs[row][feature_begin + feature] : 0.0f;
            const int attrs = heatmap_attr(runtime, value, vmax);
            if (attrs != 0) wattron(win, attrs);
            mvwaddnstr(win, y, x, "  ", 2);
            if (attrs != 0) wattroff(win, attrs);
            x += 2;
        }
    }

    std::ostringstream feature_line;
    feature_line << "features [" << feature_begin << ":" << (feature_begin + feature_count) << "]: ";
    for (std::size_t feature = 0; feature < feature_count; ++feature) {
        if (feature != 0u) feature_line << " | ";
        if (feature_begin + feature < ui.series.browse.selected_feature_names.size()) {
            feature_line << ui.series.browse.selected_feature_names[feature_begin + feature];
        } else {
            feature_line << "g" << (feature_begin + feature);
        }
    }
    draw_clipped(win, detail_pane.y + detail_pane.h - 1, detail_pane.x, detail_pane.w, feature_line.str(), A_DIM);
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
        case screen_id::home: render_home(win, runtime, ui); break;
        case screen_id::builder: render_builder(win, runtime, ui); break;
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
    if (runtime.colors_ready) wattron(win, COLOR_PAIR(cp_border));
    mvwhline(win, 0, 0, ACS_HLINE, getmaxx(win));
    if (runtime.colors_ready) wattroff(win, COLOR_PAIR(cp_border));
    draw_clipped(win,
                 0,
                 1,
                 getmaxx(win) - 2,
                 "Keys: 1-9/0/TAB screens  arrows move  e enter  l load  o open  r convert  p preprocess  a assign  u auto  x export  c apply  q quit",
                 A_BOLD | color_attr(runtime, cp_footer));
    std::string hint;
    switch (ui.active) {
        case screen_id::home: hint = "Home is the landing screen for build, load, and inspect workflows."; break;
        case screen_id::builder: hint = "Builder uses browser, drafts, and detail focus to assemble sources without a prewritten manifest."; break;
        case screen_id::sources: hint = "Sources focus persists selection and scroll state."; break;
        case screen_id::datasets: hint = "Datasets view shows global row placement."; break;
        case screen_id::parts: hint = "Parts view shows shard assignment and byte estimates."; break;
        case screen_id::shards: hint = "Shards view shows the packed output layout."; break;
        case screen_id::output: hint = "Output fields recompute the plan after edits."; break;
        case screen_id::run: hint = "Run uses the current plan and then opens the output series."; break;
        case screen_id::inspect: hint = "Inspect browses datasets, metadata facets, and cached heatmaps without touching raw CSR on keypress."; break;
        case screen_id::preprocess: hint = "Preprocess runs on the currently open series."; break;
    }
    draw_clipped(win, 1, 1, getmaxx(win) - 2, hint, color_attr(runtime, cp_footer));
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
            if (ui->active == screen_id::inspect
                && ui->inspect_focus == inspect_focus_id::heatmap
                && ui->inspect_mode == inspect_heatmap_mode::part_samples) {
                const std::size_t part_index =
                    inspect_part_global_index(*ui, ui->inspect_dataset_selection, ui->inspect_part_selection);
                if (part_index != std::numeric_limits<std::size_t>::max()
                    && part_index + 1u < ui->series.browse.part_sample_row_offsets.size()) {
                    const std::size_t row_total =
                        (std::size_t) (ui->series.browse.part_sample_row_offsets[part_index + 1u]
                                       - ui->series.browse.part_sample_row_offsets[part_index]);
                    const int delta = next.value * std::max(1, visible_rows_for_active(*ui, layout) - 1);
                    const int next_row = std::clamp((int) ui->inspect_heatmap_row_scroll + delta,
                                                    0,
                                                    std::max(0, (int) row_total - 1));
                    ui->inspect_heatmap_row_scroll = (std::size_t) next_row;
                }
            } else if (ui->active == screen_id::inspect
                       && ui->inspect_focus == inspect_focus_id::heatmap
                       && ui->inspect_mode == inspect_heatmap_mode::shard_mean) {
                const std::size_t total = inspect_shard_count_for_dataset(*ui, ui->inspect_dataset_selection);
                if (total != 0u) {
                    const int delta = next.value * std::max(1, visible_rows_for_active(*ui, layout) - 1);
                    const int next_row = std::clamp((int) ui->inspect_shard_selection + delta, 0, (int) total - 1);
                    ui->inspect_shard_selection = (std::size_t) next_row;
                    ui->inspect_heatmap_row_scroll = ui->inspect_shard_selection;
                }
            } else {
                move_active_selection(ui,
                                      next.value * std::max(1, visible_rows_for_active(*ui, layout) - 1),
                                      visible_rows_for_active(*ui, layout));
            }
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
        case action_id::inspect_prev_focus:
            if (ui->active == screen_id::inspect) {
                ui->inspect_focus = (ui->inspect_focus == inspect_focus_id::datasets)
                                        ? inspect_focus_id::heatmap
                                        : static_cast<inspect_focus_id>(static_cast<int>(ui->inspect_focus) - 1);
            } else if (ui->active == screen_id::builder) {
                ui->builder.focus = (ui->builder.focus == builder_focus_id::browser)
                                        ? builder_focus_id::detail
                                        : static_cast<builder_focus_id>(static_cast<int>(ui->builder.focus) - 1);
            }
            break;
        case action_id::inspect_next_focus:
            if (ui->active == screen_id::inspect) {
                ui->inspect_focus = (ui->inspect_focus == inspect_focus_id::heatmap)
                                        ? inspect_focus_id::datasets
                                        : static_cast<inspect_focus_id>(static_cast<int>(ui->inspect_focus) + 1);
            } else if (ui->active == screen_id::builder) {
                ui->builder.focus = (ui->builder.focus == builder_focus_id::detail)
                                        ? builder_focus_id::browser
                                        : static_cast<builder_focus_id>(static_cast<int>(ui->builder.focus) + 1);
            }
            break;
        case action_id::inspect_cycle_mode:
            if (ui->active == screen_id::inspect) {
                ui->inspect_mode =
                    ui->inspect_mode == inspect_heatmap_mode::part_samples
                        ? inspect_heatmap_mode::dataset_mean
                        : static_cast<inspect_heatmap_mode>(static_cast<int>(ui->inspect_mode) + 1);
                ui->inspect_heatmap_row_scroll = 0;
            }
            break;
        case action_id::inspect_pan_columns:
            if (ui->active == screen_id::inspect && ui->series.browse.selected_feature_count != 0u) {
                const std::size_t limit = ui->series.browse.selected_feature_count - 1u;
                const int next_col = std::clamp((int) ui->inspect_heatmap_col_scroll + next.value, 0, (int) limit);
                ui->inspect_heatmap_col_scroll = (std::size_t) next_col;
            }
            break;
        case action_id::builder_toggle_mark:
            if (ui->active == screen_id::builder && ui->builder.focus == builder_focus_id::browser) {
                builder_toggle_mark(ui);
            } else if (ui->active == screen_id::sources && !ui->inspection.sources.empty()) {
                ui->inspection.sources[ui->selected_source].included = !ui->inspection.sources[ui->selected_source].included;
                rebuild_plan(ui);
            }
            break;
        case action_id::builder_parent_dir:
            if (ui->active == screen_id::builder) {
                const std::filesystem::path current(ui->builder.current_dir);
                if (current.has_parent_path()) builder_set_dir(ui, current.parent_path().string());
            }
            break;
        case action_id::builder_auto_group:
            if (ui->active == screen_id::builder) auto_group_builder(ui);
            break;
        case action_id::builder_cycle_role:
            if (ui->active == screen_id::builder) {
                ui->builder.active_role =
                    ui->builder.active_role == wb::builder_path_role::metadata
                        ? wb::builder_path_role::matrix
                        : static_cast<wb::builder_path_role>(static_cast<int>(ui->builder.active_role) + 1);
            }
            break;
        case action_id::builder_assign:
            if (ui->active == screen_id::builder) builder_assign_selection(ui);
            break;
        case action_id::builder_new_dataset:
            if (ui->active == screen_id::builder) {
                wb::draft_dataset draft;
                draft.dataset_id = "dataset_" + std::to_string(ui->builder.drafts.size() + 1u);
                ui->builder.drafts.push_back(std::move(draft));
                ui->builder.draft_selection = ui->builder.drafts.size() - 1u;
                ui->builder.focus = builder_focus_id::drafts;
            }
            break;
        case action_id::builder_delete_dataset:
            if (ui->active == screen_id::builder && !ui->builder.drafts.empty()) {
                ui->builder.drafts.erase(ui->builder.drafts.begin() + static_cast<std::ptrdiff_t>(ui->builder.draft_selection));
                ensure_builder_draft(ui);
            }
            break;
        case action_id::builder_export_manifest:
            if (ui->active == screen_id::builder) builder_export_manifest(ui);
            break;
        case action_id::builder_apply_manifest:
            if (ui->active == screen_id::builder) builder_apply_to_workbench(ui);
            break;
        case action_id::builder_clear_field:
            if (ui->active == screen_id::builder && ui->builder.focus == builder_focus_id::detail) {
                std::string *field = builder_selected_detail_field(ui);
                if (field != nullptr) field->clear();
            }
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
        std::error_code cwd_ec;
        ui.builder.current_dir = std::filesystem::current_path(cwd_ec).string();
        if (cwd_ec || ui.builder.current_dir.empty()) ui.builder.current_dir = ".";
        refresh_builder_entries(&ui);
        auto_group_builder(&ui);
        ensure_builder_draft(&ui);
        push_log(&ui, log_level::note, "home now starts with a manifest builder and brighter landing screen");
        push_log(&ui, log_level::note, "builder can auto-group MTX triplets, generate sources in memory, and export TSV");

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
