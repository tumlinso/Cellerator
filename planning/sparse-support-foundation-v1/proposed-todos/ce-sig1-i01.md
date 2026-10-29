# CE-SIG1-I01: Implement exact support, index, and condition primitives

Implement identity-preserving support storage, reverse dependency postings, sorted support/membership intersections, temporal occurrence lists, multi-condition programs, cell condition bitmaps, and separate forward/backward work construction. Retrieval must return variable-size results with explicit capacities and overflow reporting, never silent truncation. Preserve event repetitions and ordered dependency participants. Keep proposal generation separate from exact eligibility.

Acceptance: CPU reference comparisons cover empty/all-active sets, repeated participants/events, skewed degrees, windows and expiration, multi-condition gating, and capacity exhaustion. No exhaustive pairwise discovery is introduced.
