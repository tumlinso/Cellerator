#!/usr/bin/env python3
"""Exhaustive host check of the ordered/disjoint concatenation precondition.

This is a semantic property check, not an execution model for CUDA kernels.
Each posting list represents the canonical host projection's sorted edge IDs.
"""

from itertools import combinations, product


def ordered_unique(keys):
    return all(keys[i - 1] < keys[i] for i in range(1, len(keys)))


def serial_precondition(keys, postings):
    if not ordered_unique(keys):
        return False
    previous_last = None
    for key in keys:
        row = postings[key]
        if row and previous_last is not None and row[0] <= previous_last:
            return False
        if row:
            previous_last = row[-1]
    return True


def parallel_precondition(keys, postings):
    # The large-batch CTA specialization additionally rejects empty ranges.
    return bool(keys) and all(postings[key] for key in keys) and serial_precondition(
        keys, postings)


def union(keys, postings):
    return sorted({edge for key in keys for edge in postings[key]})


def main():
    terms = [(0, entity) for entity in range(3)]
    subsets = [tuple(combo) for size in range(5)
               for combo in combinations(range(4), size)]
    rows = [()] + [(x,) for x in terms]
    rows += [(x, y) for x in terms for y in terms]
    rows += [(x, y, z) for x in terms for y in terms for z in terms]
    checked = 0
    serial_fast = 0
    parallel_fast = 0
    fallback = 0
    for a, b, c in product(subsets, repeat=3):
        postings = dict(zip(terms, (a, b, c)))
        for keys in rows:
            expected = union(keys, postings)
            if serial_precondition(keys, postings):
                concatenated = [edge for key in keys for edge in postings[key]]
                assert concatenated == expected, (keys, postings, concatenated, expected)
                serial_fast += 1
            else:
                fallback += 1
            if parallel_precondition(keys, postings):
                concatenated = [edge for key in keys for edge in postings[key]]
                assert concatenated == expected, (keys, postings, concatenated, expected)
                parallel_fast += 1
            checked += 1

    # Boundary arithmetic for the CTA's 128 * 16 term scan. Exactly 2048
    # terms are covered; 2049 must select merge fallback before scanning.
    block_capacity = 128 * 16
    for count in (0, 1, 15, 16, 17, 127, 128, 129, 2047, 2048, 2049):
        eligible_size = count <= block_capacity
        assert eligible_size == (count <= 2048)
        covered = sum(min(16, max(0, count - lane * 16)) for lane in range(128))
        assert covered == min(count, block_capacity)
        if eligible_size:
            lengths = [1 + ((index * 7) % 17) for index in range(count)]
            thread_sums = [sum(lengths[lane * 16:(lane + 1) * 16])
                           for lane in range(128)]
            thread_prefixes = []
            total = 0
            for thread_sum in thread_sums:
                thread_prefixes.append(total)
                total += thread_sum
            prefixes = []
            for lane in range(128):
                local_prefix = 0
                for local_index in range(16):
                    index = lane * 16 + local_index
                    if index >= count:
                        break
                    prefixes.append(thread_prefixes[lane] + local_prefix)
                    local_prefix += lengths[index]
            expected_prefixes = []
            total = 0
            for length in lengths:
                expected_prefixes.append(total)
                total += length
            assert prefixes == expected_prefixes

    print({
        "posting_list_assignments": len(subsets) ** len(terms),
        "query_rows_per_assignment": len(rows),
        "rows_checked": checked,
        "serial_concat_rows": serial_fast,
        "parallel_concat_rows": parallel_fast,
        "fallback_rows": fallback,
        "scan_capacity_terms": block_capacity,
        "cta_prefix_cases": [0, 1, 15, 16, 17, 127, 128, 129, 2047, 2048],
        "boundaries_checked": [0, 1, 15, 16, 17, 127, 128, 129, 2047, 2048, 2049],
        "result": "pass",
    })


if __name__ == "__main__":
    main()
