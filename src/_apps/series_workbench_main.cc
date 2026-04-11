#include "series_workbench.hh"

#include <ncurses.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
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

struct ui_state {
    screen_id active = screen_id::sources;
    wb::manifest_inspection inspection;
    wb::ingest_plan plan;
    wb::series_summary series;
    wb::conversion_report conversion;
    wb::preprocess_summary preprocess_run;
    wb::ingest_policy policy;
    wb::preprocess_config preprocess;
    std::vector<std::string> log_lines;
    std::size_t selected_source = 0;
    std::size_t selected_dataset = 0;
    std::size_t selected_part = 0;
    std::size_t selected_shard = 0;
    int output_field = 0;
    int preprocess_field = 0;
};

constexpr const char *screen_names[] = {
    "Sources", "Datasets", "Parts", "Shards", "Output", "Run", "Inspect", "Preprocess"
};

void push_log(ui_state *ui, const std::string &message) {
    if (ui == nullptr) return;
    ui->log_lines.push_back(message);
    if (ui->log_lines.size() > 128u) {
        ui->log_lines.erase(ui->log_lines.begin(), ui->log_lines.begin() + 32);
    }
}

std::string trim_copy(std::string value) {
    while (!value.empty() && std::isspace((unsigned char) value.front())) value.erase(value.begin());
    while (!value.empty() && std::isspace((unsigned char) value.back())) value.pop_back();
    return value;
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
}

void rebuild_plan(ui_state *ui) {
    if (ui == nullptr) return;
    ui->plan = wb::plan_series_ingest(ui->inspection.sources, ui->policy);
    clamp_selection(ui);
    std::ostringstream oss;
    oss << "planned " << ui->plan.datasets.size()
        << " dataset(s), " << ui->plan.parts.size()
        << " part(s), " << ui->plan.shards.size() << " shard(s)";
    push_log(ui, oss.str());
}

void load_manifest(ui_state *ui, const std::string &path) {
    if (ui == nullptr) return;
    ui->inspection = wb::inspect_manifest_tsv(path, ui->policy.reader_bytes);
    ui->conversion = {};
    ui->series = {};
    ui->preprocess_run = {};
    if (ui->inspection.ok) push_log(ui, "loaded manifest: " + path);
    else push_log(ui, "manifest has errors: " + path);
    rebuild_plan(ui);
}

void open_series(ui_state *ui, const std::string &path) {
    if (ui == nullptr) return;
    ui->series = wb::summarize_series_csh5(path);
    if (ui->series.ok) push_log(ui, "opened series: " + path);
    else push_log(ui, "failed to inspect series: " + path);
}

void run_conversion(ui_state *ui) {
    if (ui == nullptr) return;
    push_log(ui, "starting conversion");
    ui->conversion = wb::convert_plan_to_series_csh5(ui->plan);
    for (const wb::run_event &event : ui->conversion.events) {
        push_log(ui, event.phase + ": " + event.message);
    }
    for (const wb::issue &entry : ui->conversion.issues) {
        push_log(ui, wb::severity_name(entry.severity) + " " + entry.scope + ": " + entry.message);
    }
    if (ui->conversion.ok) {
        open_series(ui, ui->policy.output_path);
        push_log(ui, "conversion completed");
    } else {
        push_log(ui, "conversion failed");
    }
}

void run_preprocess(ui_state *ui) {
    if (ui == nullptr) return;
    if (ui->series.path.empty()) {
        push_log(ui, "no series.csh5 is open");
        return;
    }
    push_log(ui, "starting preprocess");
    ui->preprocess_run = wb::run_preprocess_pass(ui->series.path, ui->preprocess);
    for (const wb::issue &entry : ui->preprocess_run.issues) {
        push_log(ui, wb::severity_name(entry.severity) + " " + entry.scope + ": " + entry.message);
    }
    if (ui->preprocess_run.ok) {
        std::ostringstream oss;
        oss << "preprocess kept_cells=" << ui->preprocess_run.kept_cells
            << " kept_genes=" << ui->preprocess_run.kept_genes;
        push_log(ui, oss.str());
    } else {
        push_log(ui, "preprocess failed");
    }
}

std::string prompt_string(const std::string &label, const std::string &initial = std::string()) {
    char buffer[4096];
    std::snprintf(buffer, sizeof(buffer), "%s", initial.c_str());
    echo();
    curs_set(1);
    mvprintw(LINES - 1, 0, "%s", std::string(COLS, ' ').c_str());
    mvprintw(LINES - 1, 0, "%s", label.c_str());
    clrtoeol();
    getnstr(buffer, (int) sizeof(buffer) - 1);
    noecho();
    curs_set(0);
    return trim_copy(buffer);
}

unsigned long prompt_u64(const std::string &label, unsigned long current) {
    const std::string text = prompt_string(label, std::to_string(current));
    if (text.empty()) return current;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != 0) return current;
    return (unsigned long) parsed;
}

float prompt_f32(const std::string &label, float current) {
    const std::string text = prompt_string(label, std::to_string(current));
    if (text.empty()) return current;
    char *end = nullptr;
    const float parsed = std::strtof(text.c_str(), &end);
    if (end == text.c_str() || *end != 0) return current;
    return parsed;
}

int prompt_i32(const std::string &label, int current) {
    const std::string text = prompt_string(label, std::to_string(current));
    if (text.empty()) return current;
    char *end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != 0) return current;
    return (int) parsed;
}

void draw_boxed_window(WINDOW *win, const char *title) {
    if (win == nullptr) return;
    box(win, 0, 0);
    if (title != nullptr) mvwprintw(win, 0, 2, " %s ", title);
}

void render_nav(WINDOW *win, const ui_state &ui) {
    draw_boxed_window(win, "Views");
    for (int i = 0; i < 8; ++i) {
        if ((int) ui.active == i) wattron(win, A_REVERSE);
        mvwprintw(win, 2 + i, 2, "%d %s", i + 1, screen_names[i]);
        if ((int) ui.active == i) wattroff(win, A_REVERSE);
    }
    mvwprintw(win, 12, 2, "l load manifest");
    mvwprintw(win, 13, 2, "o open series");
    mvwprintw(win, 14, 2, "r run convert");
    mvwprintw(win, 15, 2, "p preprocess");
    mvwprintw(win, 16, 2, "e edit field");
    mvwprintw(win, 17, 2, "space toggle");
    mvwprintw(win, 18, 2, "q quit");
}

void render_summary(WINDOW *win, const ui_state &ui) {
    draw_boxed_window(win, "Summary");
    mvwprintw(win, 2, 2, "plan: %s", ui.plan.ok ? "ok" : "errors");
    mvwprintw(win, 3, 2, "datasets: %zu", ui.plan.datasets.size());
    mvwprintw(win, 4, 2, "parts: %zu", ui.plan.parts.size());
    mvwprintw(win, 5, 2, "shards: %zu", ui.plan.shards.size());
    mvwprintw(win, 6, 2, "rows: %lu", ui.plan.total_rows);
    mvwprintw(win, 7, 2, "cols: %lu", ui.plan.total_cols);
    mvwprintw(win, 8, 2, "nnz: %lu", ui.plan.total_nnz);
    mvwprintw(win, 9, 2, "bytes: %.2f GiB", ui.plan.total_estimated_bytes / 1073741824.0);
    mvwprintw(win, 11, 2, "series: %s", ui.series.ok ? "open" : "none");
    if (ui.series.ok) {
        mvwprintw(win, 12, 2, "s_rows: %llu", (unsigned long long) ui.series.rows);
        mvwprintw(win, 13, 2, "s_cols: %llu", (unsigned long long) ui.series.cols);
        mvwprintw(win, 14, 2, "s_parts: %llu", (unsigned long long) ui.series.num_parts);
        mvwprintw(win, 15, 2, "s_shards: %llu", (unsigned long long) ui.series.num_shards);
    }
}

void render_sources(WINDOW *win, const ui_state &ui) {
    draw_boxed_window(win, "Sources");
    mvwprintw(win, 1, 2, "space include/exclude, e dataset id, f cycle format");
    for (int row = 0; row < getmaxy(win) - 4 && row < (int) ui.inspection.sources.size(); ++row) {
        const std::size_t index = (std::size_t) row;
        const wb::source_entry &source = ui.inspection.sources[index];
        if (index == ui.selected_source) wattron(win, A_REVERSE);
        mvwprintw(win,
                  3 + row,
                  2,
                  "[%c] %-16s %-9s rows=%-8lu cols=%-8lu nnz=%-10lu",
                  source.included ? 'x' : ' ',
                  source.dataset_id.c_str(),
                  wb::format_name(source.format).c_str(),
                  source.rows,
                  source.cols,
                  source.nnz);
        if (index == ui.selected_source) wattroff(win, A_REVERSE);
    }
}

void render_datasets(WINDOW *win, const ui_state &ui) {
    draw_boxed_window(win, "Datasets");
    for (int row = 0; row < getmaxy(win) - 3 && row < (int) ui.plan.datasets.size(); ++row) {
        const std::size_t index = (std::size_t) row;
        const wb::planned_dataset &dataset = ui.plan.datasets[index];
        if (index == ui.selected_dataset) wattron(win, A_REVERSE);
        mvwprintw(win,
                  2 + row,
                  2,
                  "%-18s rows=%-8lu cols=%-8lu nnz=%-10lu parts=%-4lu row=[%lu,%lu)",
                  dataset.dataset_id.c_str(),
                  dataset.rows,
                  dataset.cols,
                  dataset.nnz,
                  dataset.part_count,
                  dataset.global_row_begin,
                  dataset.global_row_end);
        if (index == ui.selected_dataset) wattroff(win, A_REVERSE);
    }
}

void render_parts(WINDOW *win, const ui_state &ui) {
    draw_boxed_window(win, "Parts");
    for (int row = 0; row < getmaxy(win) - 3 && row < (int) ui.plan.parts.size(); ++row) {
        const std::size_t index = (std::size_t) row;
        const wb::planned_part &part = ui.plan.parts[index];
        if (index == ui.selected_part) wattron(win, A_REVERSE);
        mvwprintw(win,
                  2 + row,
                  2,
                  "p%-4lu %-16s rows=%-6lu nnz=%-9lu shard=%-4lu row=[%lu,%lu)",
                  part.part_id,
                  part.dataset_id.c_str(),
                  part.rows,
                  part.nnz,
                  part.shard_id,
                  part.row_begin,
                  part.row_end);
        if (index == ui.selected_part) wattroff(win, A_REVERSE);
    }
}

void render_shards(WINDOW *win, const ui_state &ui) {
    draw_boxed_window(win, "Shards");
    for (int row = 0; row < getmaxy(win) - 3 && row < (int) ui.plan.shards.size(); ++row) {
        const std::size_t index = (std::size_t) row;
        const wb::planned_shard &shard = ui.plan.shards[index];
        if (index == ui.selected_shard) wattron(win, A_REVERSE);
        mvwprintw(win,
                  2 + row,
                  2,
                  "s%-4lu parts=[%lu,%lu) rows=%-8lu nnz=%-10lu bytes=%.2f MiB",
                  shard.shard_id,
                  shard.part_begin,
                  shard.part_end,
                  shard.rows,
                  shard.nnz,
                  shard.estimated_bytes / 1048576.0);
        if (index == ui.selected_shard) wattroff(win, A_REVERSE);
    }
}

void render_output(WINDOW *win, const ui_state &ui) {
    static const char *labels[] = {
        "output_path", "max_part_nnz", "max_window_bytes", "reader_bytes", "verify_after_write"
    };
    draw_boxed_window(win, "Output");
    mvwprintw(win, 1, 2, "use up/down to pick a field, e to edit");
    for (int i = 0; i < 5; ++i) {
        if (ui.output_field == i) wattron(win, A_REVERSE);
        switch (i) {
            case 0: mvwprintw(win, 3 + i, 2, "%-18s %s", labels[i], ui.policy.output_path.c_str()); break;
            case 1: mvwprintw(win, 3 + i, 2, "%-18s %lu", labels[i], ui.policy.max_part_nnz); break;
            case 2: mvwprintw(win, 3 + i, 2, "%-18s %lu", labels[i], ui.policy.max_window_bytes); break;
            case 3: mvwprintw(win, 3 + i, 2, "%-18s %zu", labels[i], ui.policy.reader_bytes); break;
            case 4: mvwprintw(win, 3 + i, 2, "%-18s %s", labels[i], ui.policy.verify_after_write ? "true" : "false"); break;
        }
        if (ui.output_field == i) wattroff(win, A_REVERSE);
    }
}

void render_run(WINDOW *win, const ui_state &ui) {
    draw_boxed_window(win, "Run");
    mvwprintw(win, 1, 2, "press r to write series.csh5");
    mvwprintw(win, 3, 2, "status: %s", ui.conversion.ok ? "ok" : "not run / failed");
    int row = 5;
    for (const wb::run_event &event : ui.conversion.events) {
        if (row >= getmaxy(win) - 2) break;
        mvwprintw(win, row++, 2, "%s: %s", event.phase.c_str(), event.message.c_str());
    }
}

void render_inspect(WINDOW *win, const ui_state &ui) {
    draw_boxed_window(win, "Inspect");
    if (!ui.series.ok) {
        mvwprintw(win, 2, 2, "open a series.csh5 with o");
        return;
    }
    mvwprintw(win, 2, 2, "path: %s", ui.series.path.c_str());
    mvwprintw(win, 3, 2, "rows=%llu cols=%llu nnz=%llu",
              (unsigned long long) ui.series.rows,
              (unsigned long long) ui.series.cols,
              (unsigned long long) ui.series.nnz);
    mvwprintw(win, 4, 2, "datasets=%llu parts=%llu shards=%llu",
              (unsigned long long) ui.series.num_datasets,
              (unsigned long long) ui.series.num_parts,
              (unsigned long long) ui.series.num_shards);
    int row = 6;
    for (const wb::series_dataset_summary &dataset : ui.series.datasets) {
        if (row >= getmaxy(win) - 2) break;
        mvwprintw(win,
                  row++,
                  2,
                  "%-16s rows=%-8llu cols=%-8llu format=%s",
                  dataset.dataset_id.c_str(),
                  (unsigned long long) dataset.rows,
                  (unsigned long long) dataset.cols,
                  wb::format_name(dataset.format).c_str());
    }
}

void render_preprocess(WINDOW *win, const ui_state &ui) {
    static const char *labels[] = {
        "target_sum", "min_counts", "min_genes", "max_mito_fraction",
        "min_gene_sum", "min_detected", "min_variance", "device", "cache_dir", "mito_prefix"
    };
    draw_boxed_window(win, "Preprocess");
    mvwprintw(win, 1, 2, "use up/down to pick a field, e to edit, p to run");
    for (int i = 0; i < 10; ++i) {
        if (ui.preprocess_field == i) wattron(win, A_REVERSE);
        switch (i) {
            case 0: mvwprintw(win, 3 + i, 2, "%-18s %.3f", labels[i], ui.preprocess.target_sum); break;
            case 1: mvwprintw(win, 3 + i, 2, "%-18s %.3f", labels[i], ui.preprocess.min_counts); break;
            case 2: mvwprintw(win, 3 + i, 2, "%-18s %u", labels[i], ui.preprocess.min_genes); break;
            case 3: mvwprintw(win, 3 + i, 2, "%-18s %.3f", labels[i], ui.preprocess.max_mito_fraction); break;
            case 4: mvwprintw(win, 3 + i, 2, "%-18s %.3f", labels[i], ui.preprocess.min_gene_sum); break;
            case 5: mvwprintw(win, 3 + i, 2, "%-18s %.3f", labels[i], ui.preprocess.min_detected_cells); break;
            case 6: mvwprintw(win, 3 + i, 2, "%-18s %.3f", labels[i], ui.preprocess.min_variance); break;
            case 7: mvwprintw(win, 3 + i, 2, "%-18s %d", labels[i], ui.preprocess.device); break;
            case 8: mvwprintw(win, 3 + i, 2, "%-18s %s", labels[i], ui.preprocess.cache_dir.c_str()); break;
            case 9: mvwprintw(win, 3 + i, 2, "%-18s %s", labels[i], ui.preprocess.mito_prefix.c_str()); break;
        }
        if (ui.preprocess_field == i) wattroff(win, A_REVERSE);
    }
    mvwprintw(win, 15, 2, "last run: %s", ui.preprocess_run.ok ? "ok" : "not run / failed");
    if (ui.preprocess_run.ok) {
        mvwprintw(win, 16, 2, "kept_cells=%.1f kept_genes=%lu", ui.preprocess_run.kept_cells, ui.preprocess_run.kept_genes);
    }
}

void render_main(WINDOW *win, const ui_state &ui) {
    switch (ui.active) {
        case screen_id::sources: render_sources(win, ui); break;
        case screen_id::datasets: render_datasets(win, ui); break;
        case screen_id::parts: render_parts(win, ui); break;
        case screen_id::shards: render_shards(win, ui); break;
        case screen_id::output: render_output(win, ui); break;
        case screen_id::run: render_run(win, ui); break;
        case screen_id::inspect: render_inspect(win, ui); break;
        case screen_id::preprocess: render_preprocess(win, ui); break;
    }
}

void render_log(WINDOW *win, const ui_state &ui) {
    draw_boxed_window(win, "Log");
    const int available = getmaxy(win) - 2;
    const int start = std::max<int>(0, (int) ui.log_lines.size() - available);
    for (int i = 0; i < available && start + i < (int) ui.log_lines.size(); ++i) {
        mvwprintw(win, 1 + i, 2, "%s", ui.log_lines[(std::size_t) start + i].c_str());
    }
}

void render_ui(const ui_state &ui) {
    const int nav_w = std::max(24, COLS / 4);
    const int summary_w = std::max(24, COLS / 5);
    const int log_h = std::max(8, LINES / 5);
    const int main_w = std::max(20, COLS - nav_w - summary_w);
    const int body_h = std::max(10, LINES - log_h - 1);

    erase();
    mvprintw(0, 0, "Cellerator Workbench  mode=%s  output=%s",
             screen_names[(int) ui.active],
             ui.policy.output_path.empty() ? "<unset>" : ui.policy.output_path.c_str());
    clrtoeol();

    WINDOW *nav = newwin(body_h, nav_w, 1, 0);
    WINDOW *main = newwin(body_h, main_w, 1, nav_w);
    WINDOW *summary = newwin(body_h, summary_w, 1, nav_w + main_w);
    WINDOW *log = newwin(log_h, COLS, 1 + body_h, 0);

    render_nav(nav, ui);
    render_main(main, ui);
    render_summary(summary, ui);
    render_log(log, ui);

    wrefresh(nav);
    wrefresh(main);
    wrefresh(summary);
    wrefresh(log);
    delwin(nav);
    delwin(main);
    delwin(summary);
    delwin(log);
    refresh();
}

void edit_active_field(ui_state *ui) {
    if (ui == nullptr) return;
    switch (ui->active) {
        case screen_id::sources:
            if (!ui->inspection.sources.empty()) {
                wb::source_entry &source = ui->inspection.sources[ui->selected_source];
                source.dataset_id = prompt_string("dataset_id: ", source.dataset_id);
                rebuild_plan(ui);
            }
            break;
        case screen_id::output:
            switch (ui->output_field) {
                case 0: ui->policy.output_path = prompt_string("output_path: ", ui->policy.output_path); break;
                case 1: ui->policy.max_part_nnz = prompt_u64("max_part_nnz: ", ui->policy.max_part_nnz); break;
                case 2: ui->policy.max_window_bytes = prompt_u64("max_window_bytes: ", ui->policy.max_window_bytes); break;
                case 3: ui->policy.reader_bytes = (std::size_t) prompt_u64("reader_bytes: ", (unsigned long) ui->policy.reader_bytes); break;
                case 4: ui->policy.verify_after_write = !ui->policy.verify_after_write; break;
            }
            rebuild_plan(ui);
            break;
        case screen_id::preprocess:
            switch (ui->preprocess_field) {
                case 0: ui->preprocess.target_sum = prompt_f32("target_sum: ", ui->preprocess.target_sum); break;
                case 1: ui->preprocess.min_counts = prompt_f32("min_counts: ", ui->preprocess.min_counts); break;
                case 2: ui->preprocess.min_genes = (unsigned int) prompt_u64("min_genes: ", ui->preprocess.min_genes); break;
                case 3: ui->preprocess.max_mito_fraction = prompt_f32("max_mito_fraction: ", ui->preprocess.max_mito_fraction); break;
                case 4: ui->preprocess.min_gene_sum = prompt_f32("min_gene_sum: ", ui->preprocess.min_gene_sum); break;
                case 5: ui->preprocess.min_detected_cells = prompt_f32("min_detected_cells: ", ui->preprocess.min_detected_cells); break;
                case 6: ui->preprocess.min_variance = prompt_f32("min_variance: ", ui->preprocess.min_variance); break;
                case 7: ui->preprocess.device = prompt_i32("device: ", ui->preprocess.device); break;
                case 8: ui->preprocess.cache_dir = prompt_string("cache_dir: ", ui->preprocess.cache_dir); break;
                case 9: ui->preprocess.mito_prefix = prompt_string("mito_prefix: ", ui->preprocess.mito_prefix); break;
            }
            break;
        default:
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

void move_selection(ui_state *ui, int delta) {
    if (ui == nullptr) return;
    switch (ui->active) {
        case screen_id::sources:
            if (!ui->inspection.sources.empty()) {
                const int next = std::clamp<int>((int) ui->selected_source + delta, 0, (int) ui->inspection.sources.size() - 1);
                ui->selected_source = (std::size_t) next;
            }
            break;
        case screen_id::datasets:
            if (!ui->plan.datasets.empty()) {
                const int next = std::clamp<int>((int) ui->selected_dataset + delta, 0, (int) ui->plan.datasets.size() - 1);
                ui->selected_dataset = (std::size_t) next;
            }
            break;
        case screen_id::parts:
            if (!ui->plan.parts.empty()) {
                const int next = std::clamp<int>((int) ui->selected_part + delta, 0, (int) ui->plan.parts.size() - 1);
                ui->selected_part = (std::size_t) next;
            }
            break;
        case screen_id::shards:
            if (!ui->plan.shards.empty()) {
                const int next = std::clamp<int>((int) ui->selected_shard + delta, 0, (int) ui->plan.shards.size() - 1);
                ui->selected_shard = (std::size_t) next;
            }
            break;
        case screen_id::output:
            ui->output_field = std::clamp(ui->output_field + delta, 0, 4);
            break;
        case screen_id::preprocess:
            ui->preprocess_field = std::clamp(ui->preprocess_field + delta, 0, 9);
            break;
        default:
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
                (unsigned long long) summary.rows,
                (unsigned long long) summary.cols,
                (unsigned long long) summary.nnz,
                (unsigned long long) summary.num_datasets,
                (unsigned long long) summary.num_parts,
                (unsigned long long) summary.num_shards);
    for (const wb::series_dataset_summary &dataset : summary.datasets) {
        std::printf("dataset %s rows=%llu cols=%llu nnz=%llu format=%s\n",
                    dataset.dataset_id.c_str(),
                    (unsigned long long) dataset.rows,
                    (unsigned long long) dataset.cols,
                    (unsigned long long) dataset.nnz,
                    wb::format_name(dataset.format).c_str());
    }
}

} // namespace

int main(int argc, char **argv) {
    ui_state ui;
    std::string manifest_to_load;
    std::string series_to_open;
    bool dump_plan_only = false;
    bool dump_series_only = false;

    ui.policy.output_path = "series.csh5";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--manifest" || arg == "-m") && i + 1 < argc) {
            manifest_to_load = argv[++i];
        } else if ((arg == "--open" || arg == "-o") && i + 1 < argc) {
            series_to_open = argv[++i];
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

    if (!manifest_to_load.empty()) load_manifest(&ui, manifest_to_load);
    if (!series_to_open.empty()) open_series(&ui, series_to_open);

    if (dump_plan_only) {
        dump_plan(ui.plan);
        return ui.plan.ok ? 0 : 1;
    }
    if (dump_series_only) {
        dump_series(ui.series);
        return ui.series.ok ? 0 : 1;
    }

    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);

    bool running = true;
    while (running) {
        render_ui(ui);
        const int ch = getch();
        switch (ch) {
            case 'q': running = false; break;
            case '1': ui.active = screen_id::sources; break;
            case '2': ui.active = screen_id::datasets; break;
            case '3': ui.active = screen_id::parts; break;
            case '4': ui.active = screen_id::shards; break;
            case '5': ui.active = screen_id::output; break;
            case '6': ui.active = screen_id::run; break;
            case '7': ui.active = screen_id::inspect; break;
            case '8': ui.active = screen_id::preprocess; break;
            case KEY_UP:
            case 'k': move_selection(&ui, -1); break;
            case KEY_DOWN:
            case 'j': move_selection(&ui, 1); break;
            case 'e': edit_active_field(&ui); break;
            case 'f':
                if (ui.active == screen_id::sources) cycle_source_format(&ui);
                break;
            case ' ':
                if (ui.active == screen_id::sources && !ui.inspection.sources.empty()) {
                    ui.inspection.sources[ui.selected_source].included = !ui.inspection.sources[ui.selected_source].included;
                    rebuild_plan(&ui);
                }
                break;
            case 'l': {
                const std::string path = prompt_string("manifest path: ", manifest_to_load);
                if (!path.empty()) {
                    manifest_to_load = path;
                    load_manifest(&ui, path);
                }
                break;
            }
            case 'o': {
                const std::string path = prompt_string("series path: ", series_to_open);
                if (!path.empty()) {
                    series_to_open = path;
                    open_series(&ui, path);
                }
                break;
            }
            case 'r':
                run_conversion(&ui);
                break;
            case 'p':
                run_preprocess(&ui);
                break;
            default:
                break;
        }
    }

    endwin();
    return 0;
}
