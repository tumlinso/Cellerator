#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Participant {
    std::uint32_t predicate = 0;
    std::uint8_t expected = 0;
};

struct Posting {
    std::uint32_t rule = 0;
    std::uint8_t expected = 0;
};

struct RawFixture {
    std::uint32_t cells = 0;
    std::uint32_t predicates = 0;
    std::vector<std::vector<Participant>> rules;
    std::vector<std::uint8_t> values;
};

struct Tables {
    std::uint32_t predicates = 0;
    std::vector<std::uint64_t> rule_offsets;
    std::vector<Participant> participants;
    std::vector<std::uint64_t> posting_offsets;
    std::vector<Posting> postings;

    std::uint64_t rules() const { return rule_offsets.size() - 1; }
    std::uint64_t arity(std::uint32_t rule) const {
        return rule_offsets[rule + 1] - rule_offsets[rule];
    }
    std::uint64_t occurrences() const { return participants.size(); }
    std::uint64_t bytes(bool with_postings) const {
        auto result = rule_offsets.capacity() * sizeof(std::uint64_t)
                    + participants.capacity() * sizeof(Participant);
        if (with_postings)
            result += posting_offsets.capacity() * sizeof(std::uint64_t)
                    + postings.capacity() * sizeof(Posting);
        return result;
    }
};

struct Toggle {
    std::uint32_t cell = 0;
    std::uint32_t predicate = 0;
};

struct Batch {
    std::vector<Toggle> toggles;
};

struct Config {
    std::string name;
    std::uint32_t cells = 0;
    std::uint32_t rules = 0;
    std::uint32_t predicates = 0;
    std::uint32_t arity = 0;
    std::uint32_t touched_cells = 0;
    std::uint32_t toggles_per_cell = 0;
    std::uint32_t batches = 0;
    double initially_satisfied_participant_fraction = 0.5;
};

struct Metrics {
    std::uint64_t predicate_toggles = 0;
    std::uint64_t posting_visits = 0;
    std::uint64_t rule_evaluations = 0;
    std::uint64_t participant_checks = 0;
    std::uint64_t counter_updates = 0;
    std::uint64_t eligibility_updates = 0;
};

struct TimedResult {
    std::string strategy;
    double prepare_ms = 0.0;
    double update_ns_per_batch = 0.0;
    std::uint64_t persistent_bytes = 0;
    Metrics metrics;
    std::uint64_t checksum = 0;
};

struct Rng {
    std::uint64_t state;
    std::uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    std::uint32_t bounded(std::uint32_t limit) {
        return static_cast<std::uint32_t>(next() % limit);
    }
    double unit() { return static_cast<double>(next() >> 11) / 9007199254740992.0; }
};

Tables build_tables(const RawFixture &raw, bool with_postings) {
    Tables out;
    out.predicates = raw.predicates;
    out.rule_offsets.reserve(raw.rules.size() + 1);
    out.rule_offsets.push_back(0);
    std::uint64_t total = 0;
    for (const auto &rule : raw.rules) total += rule.size();
    out.participants.reserve(total);
    for (const auto &rule : raw.rules) {
        out.participants.insert(out.participants.end(), rule.begin(), rule.end());
        out.rule_offsets.push_back(out.participants.size());
    }
    if (!with_postings) return out;

    std::vector<std::uint64_t> counts(raw.predicates, 0);
    for (const auto &part : out.participants) ++counts[part.predicate];
    out.posting_offsets.resize(static_cast<std::size_t>(raw.predicates) + 1, 0);
    for (std::uint32_t p = 0; p < raw.predicates; ++p)
        out.posting_offsets[p + 1] = out.posting_offsets[p] + counts[p];
    out.postings.resize(total);
    auto cursor = out.posting_offsets;
    for (std::uint32_t r = 0; r < raw.rules.size(); ++r) {
        for (const auto &part : raw.rules[r]) {
            const auto at = cursor[part.predicate]++;
            out.postings[at] = Posting{r, part.expected};
        }
    }
    return out;
}

bool scalar_rule(const std::vector<Participant> &rule,
                 const std::uint8_t *values) {
    for (const auto &part : rule)
        if (values[part.predicate] != part.expected) return false;
    return true;
}

std::vector<std::uint8_t> scalar_oracle(const RawFixture &raw,
                                       const std::vector<std::uint8_t> &values) {
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(raw.cells) * raw.rules.size());
    for (std::uint32_t c = 0; c < raw.cells; ++c) {
        const auto *row = values.data() + static_cast<std::size_t>(c) * raw.predicates;
        for (std::uint32_t r = 0; r < raw.rules.size(); ++r)
            result[static_cast<std::size_t>(c) * raw.rules.size() + r] =
                static_cast<std::uint8_t>(scalar_rule(raw.rules[r], row));
    }
    return result;
}

std::vector<std::uint32_t> scalar_counts(const RawFixture &raw,
                                         const std::vector<std::uint8_t> &values) {
    std::vector<std::uint32_t> result(
        static_cast<std::size_t>(raw.cells) * raw.rules.size(), 0);
    for (std::uint32_t c = 0; c < raw.cells; ++c) {
        const auto *row = values.data() + static_cast<std::size_t>(c) * raw.predicates;
        for (std::uint32_t r = 0; r < raw.rules.size(); ++r)
            for (const auto &part : raw.rules[r])
                result[static_cast<std::size_t>(c) * raw.rules.size() + r] +=
                    static_cast<std::uint32_t>(row[part.predicate] == part.expected);
    }
    return result;
}

std::vector<Batch> make_batches(const Config &config, std::uint64_t seed) {
    Rng rng{seed};
    std::vector<Batch> batches(config.batches);
    for (std::uint32_t b = 0; b < config.batches; ++b) {
        auto &batch = batches[b];
        batch.toggles.reserve(config.touched_cells * config.toggles_per_cell);
        const auto first_cell = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(b) * config.touched_cells) % config.cells);
        for (std::uint32_t i = 0; i < config.touched_cells; ++i) {
            const auto cell = (first_cell + i) % config.cells;
            for (std::uint32_t j = 0; j < config.toggles_per_cell; ++j)
                batch.toggles.push_back(Toggle{cell, rng.bounded(config.predicates)});
        }
        std::stable_sort(batch.toggles.begin(), batch.toggles.end(),
                         [](const Toggle &a, const Toggle &b) { return a.cell < b.cell; });
    }
    return batches;
}

RawFixture make_fixture(const Config &config, std::uint64_t seed) {
    RawFixture raw;
    raw.cells = config.cells;
    raw.predicates = config.predicates;
    raw.rules.resize(config.rules);
    raw.values.assign(static_cast<std::size_t>(config.cells) * config.predicates, 0);
    Rng rng{seed};
    for (auto &rule : raw.rules) {
        rule.reserve(config.arity);
        for (std::uint32_t i = 0; i < config.arity; ++i) {
            const auto predicate = rng.bounded(config.predicates);
            const auto expected = static_cast<std::uint8_t>(
                rng.unit() < config.initially_satisfied_participant_fraction ? 0 : 1);
            rule.push_back(Participant{predicate, expected});
        }
    }
    return raw;
}

struct EngineBase {
    Tables tables;
    std::uint32_t cells = 0;
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> eligible;

    EngineBase(const RawFixture &raw, bool with_postings)
        : tables(build_tables(raw, with_postings)), cells(raw.cells), values(raw.values) {
        eligible = scalar_oracle(raw, values);
    }

    std::uint64_t common_bytes() const {
        return tables.bytes(!tables.posting_offsets.empty())
             + values.capacity() * sizeof(std::uint8_t)
             + eligible.capacity() * sizeof(std::uint8_t);
    }

    void apply_toggles(const Batch &batch, std::size_t first, std::size_t last,
                       std::uint32_t cell, Metrics &m) {
        const auto row = static_cast<std::size_t>(cell) * tables.predicates;
        for (auto i = first; i < last; ++i) {
            const auto p = batch.toggles[i].predicate;
            values[row + p] ^= 1;
            ++m.predicate_toggles;
        }
    }

    std::uint64_t checksum() const {
        std::uint64_t sum = 0;
        for (std::size_t i = 0; i < eligible.size(); ++i)
            sum += static_cast<std::uint64_t>(eligible[i]) * (i + 1);
        return sum;
    }
};

struct FullScan : EngineBase {
    explicit FullScan(const RawFixture &raw) : EngineBase(raw, false) {}

    Metrics update(const std::vector<Batch> &batches, std::uint32_t repeat) {
        Metrics m;
        const auto rules = tables.rules();
        for (std::uint32_t rep = 0; rep < repeat; ++rep) {
            for (const auto &batch : batches) {
                for (std::size_t i = 0; i < batch.toggles.size();) {
                    const auto cell = batch.toggles[i].cell;
                    auto end = i + 1;
                    while (end < batch.toggles.size() && batch.toggles[end].cell == cell) ++end;
                    apply_toggles(batch, i, end, cell, m);
                    const auto row = static_cast<std::size_t>(cell) * rules;
                    const auto *values_row = values.data()
                        + static_cast<std::size_t>(cell) * tables.predicates;
                    for (std::uint32_t r = 0; r < rules; ++r) {
                        bool ok = true;
                        ++m.rule_evaluations;
                        for (auto j = tables.rule_offsets[r]; j < tables.rule_offsets[r + 1]; ++j) {
                            ++m.participant_checks;
                            const auto &part = tables.participants[j];
                            if (values_row[part.predicate] != part.expected) {
                                ok = false;
                                break;
                            }
                        }
                        eligible[row + r] = static_cast<std::uint8_t>(ok);
                    }
                    i = end;
                }
            }
        }
        return m;
    }
};

struct DirtyRuleRecompute : EngineBase {
    std::vector<std::uint32_t> stamps;
    std::vector<std::uint32_t> dirty_rules;
    std::uint32_t epoch = 0;

    explicit DirtyRuleRecompute(const RawFixture &raw) : EngineBase(raw, true) {
        stamps.assign(tables.rules(), 0);
        dirty_rules.reserve(tables.rules());
    }

    std::uint64_t bytes() const {
        return common_bytes() + stamps.capacity() * sizeof(std::uint32_t)
             + dirty_rules.capacity() * sizeof(std::uint32_t);
    }

    Metrics update(const std::vector<Batch> &batches, std::uint32_t repeat) {
        Metrics m;
        for (std::uint32_t rep = 0; rep < repeat; ++rep) {
            for (const auto &batch : batches) {
                for (std::size_t i = 0; i < batch.toggles.size();) {
                    const auto cell = batch.toggles[i].cell;
                    auto end = i + 1;
                    while (end < batch.toggles.size() && batch.toggles[end].cell == cell) ++end;
                    apply_toggles(batch, i, end, cell, m);
                    if (++epoch == 0) {
                        std::fill(stamps.begin(), stamps.end(), 0);
                        epoch = 1;
                    }
                    dirty_rules.clear();
                    for (auto k = i; k < end; ++k) {
                        const auto p = batch.toggles[k].predicate;
                        for (auto at = tables.posting_offsets[p];
                             at < tables.posting_offsets[p + 1]; ++at) {
                            ++m.posting_visits;
                            const auto r = tables.postings[at].rule;
                            if (stamps[r] != epoch) {
                                stamps[r] = epoch;
                                dirty_rules.push_back(r);
                            }
                        }
                    }
                    const auto output_row = static_cast<std::size_t>(cell) * tables.rules();
                    const auto *values_row = values.data()
                        + static_cast<std::size_t>(cell) * tables.predicates;
                    for (const auto r : dirty_rules) {
                        ++m.rule_evaluations;
                        bool ok = true;
                        for (auto j = tables.rule_offsets[r]; j < tables.rule_offsets[r + 1]; ++j) {
                            ++m.participant_checks;
                            const auto &part = tables.participants[j];
                            if (values_row[part.predicate] != part.expected) {
                                ok = false;
                                break;
                            }
                        }
                        eligible[output_row + r] = static_cast<std::uint8_t>(ok);
                        ++m.eligibility_updates;
                    }
                    i = end;
                }
            }
        }
        return m;
    }
};

struct Counted : EngineBase {
    std::vector<std::uint32_t> satisfied;
    std::vector<std::uint32_t> stamps;
    std::vector<std::uint32_t> touched_rules;
    std::uint32_t epoch = 0;

    explicit Counted(const RawFixture &raw) : EngineBase(raw, true) {
        const auto rules = tables.rules();
        satisfied.assign(static_cast<std::size_t>(cells) * rules, 0);
        for (std::uint32_t c = 0; c < cells; ++c) {
            const auto *row = values.data() + static_cast<std::size_t>(c) * tables.predicates;
            for (std::uint32_t r = 0; r < rules; ++r)
                for (auto j = tables.rule_offsets[r]; j < tables.rule_offsets[r + 1]; ++j) {
                    const auto &part = tables.participants[j];
                    satisfied[static_cast<std::size_t>(c) * rules + r] +=
                        static_cast<std::uint32_t>(row[part.predicate] == part.expected);
                }
        }
        stamps.assign(rules, 0);
        touched_rules.reserve(rules);
    }

    std::uint64_t bytes() const {
        return common_bytes() + satisfied.capacity() * sizeof(std::uint32_t)
             + stamps.capacity() * sizeof(std::uint32_t)
             + touched_rules.capacity() * sizeof(std::uint32_t);
    }

    Metrics update(const std::vector<Batch> &batches, std::uint32_t repeat) {
        Metrics m;
        const auto rules = tables.rules();
        for (std::uint32_t rep = 0; rep < repeat; ++rep) {
            for (const auto &batch : batches) {
                for (std::size_t i = 0; i < batch.toggles.size();) {
                    const auto cell = batch.toggles[i].cell;
                    auto end = i + 1;
                    while (end < batch.toggles.size() && batch.toggles[end].cell == cell) ++end;
                    if (++epoch == 0) {
                        std::fill(stamps.begin(), stamps.end(), 0);
                        epoch = 1;
                    }
                    touched_rules.clear();
                    const auto predicate_row = static_cast<std::size_t>(cell) * tables.predicates;
                    for (auto k = i; k < end; ++k) {
                        const auto p = batch.toggles[k].predicate;
                        values[predicate_row + p] ^= 1;
                        const auto new_value = values[predicate_row + p];
                        ++m.predicate_toggles;
                        for (auto at = tables.posting_offsets[p];
                             at < tables.posting_offsets[p + 1]; ++at) {
                            ++m.posting_visits;
                            const auto &posting = tables.postings[at];
                            const auto slot = static_cast<std::size_t>(cell) * rules + posting.rule;
                            if (posting.expected == new_value) ++satisfied[slot];
                            else --satisfied[slot];
                            ++m.counter_updates;
                            if (stamps[posting.rule] != epoch) {
                                stamps[posting.rule] = epoch;
                                touched_rules.push_back(posting.rule);
                            }
                        }
                    }
                    const auto output_row = static_cast<std::size_t>(cell) * rules;
                    for (const auto r : touched_rules) {
                        eligible[output_row + r] = static_cast<std::uint8_t>(
                            satisfied[output_row + r] == tables.arity(r));
                        ++m.eligibility_updates;
                    }
                    i = end;
                }
            }
        }
        return m;
    }
};

std::uint64_t workload_bytes(const std::vector<Batch> &batches) {
    std::uint64_t bytes = batches.capacity() * sizeof(Batch);
    for (const auto &batch : batches) bytes += batch.toggles.capacity() * sizeof(Toggle);
    return bytes;
}

template <class Engine>
TimedResult time_strategy(const std::string &name, const RawFixture &raw,
                          const std::vector<Batch> &batches,
                          std::uint32_t repeat, std::uint32_t trials) {
    std::vector<double> prep_ms;
    std::vector<double> update_ns;
    TimedResult last;
    last.strategy = name;
    for (std::uint32_t trial = 0; trial < trials; ++trial) {
        const auto prepare_start = Clock::now();
        Engine engine(raw);
        const auto prepare_end = Clock::now();
        const auto update_start = Clock::now();
        last.metrics = engine.update(batches, repeat);
        const auto update_end = Clock::now();
        prep_ms.push_back(std::chrono::duration<double, std::milli>(
            prepare_end - prepare_start).count());
        update_ns.push_back(std::chrono::duration<double, std::nano>(
            update_end - update_start).count()
            / (static_cast<double>(repeat) * batches.size()));
        if constexpr (std::is_same<Engine, FullScan>::value)
            last.persistent_bytes = engine.common_bytes();
        else
            last.persistent_bytes = engine.bytes();
        last.checksum = engine.checksum();
    }
    std::sort(prep_ms.begin(), prep_ms.end());
    std::sort(update_ns.begin(), update_ns.end());
    last.prepare_ms = prep_ms[prep_ms.size() / 2];
    last.update_ns_per_batch = update_ns[update_ns.size() / 2];
    return last;
}

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

void compare_states(const RawFixture &raw, const FullScan &full,
                    const DirtyRuleRecompute &dirty, const Counted &counted) {
    const auto expected = scalar_oracle(raw, full.values);
    require(full.eligible == expected, "full scan diverged from scalar oracle");
    require(dirty.values == full.values, "dirty predicate state diverged");
    require(counted.values == full.values, "counted predicate state diverged");
    require(dirty.eligible == expected, "dirty recompute diverged from scalar oracle");
    require(counted.eligible == expected, "counted eligibility diverged from scalar oracle");
    const auto expected_counts = scalar_counts(raw, full.values);
    require(counted.satisfied == expected_counts, "counted prerequisite counters diverged");
}

void self_check() {
    RawFixture raw;
    raw.cells = 3;
    raw.predicates = 5;
    raw.rules = {
        {},                                      // Empty all-of is true.
        {{0, 1}},                                // Positive participant.
        {{0, 1}, {0, 1}},                       // Repeated participant.
        {{1, 0}, {2, 1}, {1, 0}},               // Negative and repeated participants.
        {{3, 1}, {3, 0}},                       // Contradictory overlap is never true.
        {{0, 0}, {2, 0}, {4, 0}},               // Initially all satisfied (all-true).
        {{4, 1}, {0, 0}, {4, 1}},               // Mixed polarity and repetition.
        {},                                      // Second empty rule.
    };
    raw.values = {
        0, 0, 0, 0, 0,
        1, 0, 1, 1, 0,
        0, 1, 0, 1, 1,
    };
    FullScan full(raw);
    DirtyRuleRecompute dirty(raw);
    Counted counted(raw);
    compare_states(raw, full, dirty, counted);

    const std::vector<Batch> batches = {
        Batch{{{0, 0}, {0, 0}, {0, 2}, {1, 1}, {1, 2}, {2, 4}}},
        Batch{{{0, 1}, {0, 3}, {1, 0}, {1, 0}, {2, 1}, {2, 2}}},
        Batch{{{0, 4}, {1, 4}, {2, 0}, {2, 0}, {2, 0}}},
        Batch{{{0, 2}, {0, 2}, {1, 3}, {1, 4}, {2, 1}}},
    };
    for (const auto &batch : batches) {
        full.update(std::vector<Batch>{batch}, 1);
        dirty.update(std::vector<Batch>{batch}, 1);
        counted.update(std::vector<Batch>{batch}, 1);
        compare_states(raw, full, dirty, counted);
    }

    Rng rng{0x9e3779b97f4a7c15ULL};
    RawFixture generated;
    generated.cells = 7;
    generated.predicates = 13;
    generated.rules.resize(29);
    generated.values.resize(static_cast<std::size_t>(generated.cells) * generated.predicates);
    for (auto &value : generated.values) value = static_cast<std::uint8_t>(rng.bounded(2));
    for (auto &rule : generated.rules) {
        const auto length = rng.bounded(10);
        for (std::uint32_t i = 0; i < length; ++i)
            rule.push_back(Participant{rng.bounded(generated.predicates),
                                       static_cast<std::uint8_t>(rng.bounded(2))});
    }
    FullScan generated_full(generated);
    DirtyRuleRecompute generated_dirty(generated);
    Counted generated_counted(generated);
    compare_states(generated, generated_full, generated_dirty, generated_counted);
    for (std::uint32_t step = 0; step < 100; ++step) {
        Batch batch;
        const auto n = 1 + rng.bounded(18);
        for (std::uint32_t i = 0; i < n; ++i)
            batch.toggles.push_back(Toggle{rng.bounded(generated.cells),
                                           rng.bounded(generated.predicates)});
        std::stable_sort(batch.toggles.begin(), batch.toggles.end(),
                         [](const Toggle &a, const Toggle &b) { return a.cell < b.cell; });
        const std::vector<Batch> one{batch};
        generated_full.update(one, 1);
        generated_dirty.update(one, 1);
        generated_counted.update(one, 1);
        compare_states(generated, generated_full, generated_dirty, generated_counted);
    }
    std::cout << "{\"record\":\"self_check\",\"status\":\"pass\","
                 "\"focused_rules\":8,\"random_rules\":29,\"random_cells\":7,"
                 "\"random_batches\":100,\"cases\":[\"overlap\",\"repeated_participant\","
                 "\"negative_participant\",\"empty_all_of\",\"all_true\","
                 "\"contradictory_participants\",\"multiple_toggles_per_batch\"]}\n";
}

std::vector<Config> suite() {
    return {
        {"sparse_incidence", 64, 1024, 8192, 12, 8, 1, 200, 0.50},
        {"shared_high_pass_long", 32, 1024, 256, 16, 4, 1, 200, 0.95},
        {"shared_mixed_pass_long", 32, 1024, 256, 16, 4, 1, 200, 0.50},
        {"shared_high_pass_batched", 32, 1024, 256, 16, 4, 6, 200, 0.95},
        {"shared_high_pass_short", 32, 1024, 256, 2, 4, 1, 200, 0.95},
        {"shared_dense_toggle_batch", 32, 512, 64, 16, 2, 64, 80, 0.50},
    };
}

void print_metrics(const Metrics &m) {
    std::cout << "\"predicate_toggles\":" << m.predicate_toggles
              << ",\"posting_visits\":" << m.posting_visits
              << ",\"rule_evaluations\":" << m.rule_evaluations
              << ",\"participant_checks\":" << m.participant_checks
              << ",\"counter_updates\":" << m.counter_updates
              << ",\"eligibility_updates\":" << m.eligibility_updates;
}

void run_suite() {
    constexpr std::uint32_t trials = 5;
    constexpr std::uint32_t repeats = 3;
    std::cout << std::setprecision(8);
    for (std::size_t ix = 0; ix < suite().size(); ++ix) {
        const auto config = suite()[ix];
        const auto raw = make_fixture(config, 0x123456789abcdefULL + ix * 97);
        const auto batches = make_batches(config, 0x5eedbeef123ULL + ix * 313);
        const auto base = time_strategy<FullScan>("full_scan", raw, batches, repeats, trials);
        const auto dirty = time_strategy<DirtyRuleRecompute>("dirty_recompute", raw, batches,
                                                             repeats, trials);
        const auto counted = time_strategy<Counted>("counted", raw, batches, repeats, trials);
        for (const auto *r : {&base, &dirty, &counted}) {
            std::cout << "{\"record\":\"result\",\"fixture\":\"" << config.name
                      << "\",\"strategy\":\"" << r->strategy
                      << "\",\"cells\":" << config.cells
                      << ",\"rules\":" << config.rules
                      << ",\"predicates\":" << config.predicates
                      << ",\"arity\":" << config.arity
                      << ",\"initial_satisfaction_fraction\":"
                      << config.initially_satisfied_participant_fraction
                      << ",\"touched_cells_per_batch\":" << config.touched_cells
                      << ",\"toggles_per_touched_cell\":" << config.toggles_per_cell
                      << ",\"batches\":" << config.batches
                      << ",\"timed_repeats\":" << repeats
                      << ",\"trials\":" << trials
                      << ",\"prepare_median_ms\":" << r->prepare_ms
                      << ",\"update_median_ns_per_batch\":" << r->update_ns_per_batch
                      << ",\"persistent_payload_bytes\":" << r->persistent_bytes
                      << ",\"workload_payload_bytes_shared\":" << workload_bytes(batches)
                      << ",\"checksum\":" << r->checksum << ',';
            print_metrics(r->metrics);
            std::cout << "}\n";
        }
        require(base.checksum == dirty.checksum && base.checksum == counted.checksum,
                "timed strategies produced different final checksums");
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "--suite";
        if (mode == "--check-only") {
            self_check();
        } else if (mode == "--suite") {
            self_check();
            run_suite();
        } else {
            throw std::runtime_error("usage: counted_condition_probe [--check-only|--suite]");
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "counted_condition_probe: " << error.what() << '\n';
        return 1;
    }
}
