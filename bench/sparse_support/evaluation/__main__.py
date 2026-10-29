"""CPU-only manifest/catalog front door for sparse-support evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .catalog import mechanism_catalog
from .inputs import CapacityLimits, prepare_events, prepare_ranked, write_raw_bundle


def _mapping(value):
    return {int(key): row for key, row in value.items()} if value is not None else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("catalog", help="print retained indexing/gating families")
    for mode in ("ranked", "events"):
        sub = subparsers.add_parser(f"prepare-{mode}", help=f"prepare a {mode} input bundle")
        sub.add_argument("--input-npz", required=True, type=Path,
                         help="already-tokenized arrays; loaded with allow_pickle=False")
        sub.add_argument("--spec-json", required=True, type=Path,
                         help="JSON topology, condition, window, capacity and provenance metadata")
        sub.add_argument("--output", required=True, type=Path,
                         help="new or empty directory for manifest.json and raw arrays")
    args = parser.parse_args(argv)
    if args.command == "catalog":
        print(json.dumps(mechanism_catalog(), sort_keys=True, indent=2))
        return 0

    spec = json.loads(args.spec_json.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        parser.error("spec JSON must be an object")
    with np.load(args.input_npz, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    topology = {
        "edge_ids": arrays.pop("edge_ids"),
        "source_entity_ids": arrays.pop("source_entity_ids"),
        "destination_entity_ids": arrays.pop("destination_entity_ids"),
    }
    cap_spec = spec.pop("capacities", {})
    capacities = CapacityLimits(**cap_spec)
    common = dict(
        **topology,
        cell_id_universe=arrays.pop("cell_id_universe", None),
        query_role=spec.pop("query_role", "source"),
        conditions=_mapping(spec.pop("conditions", None)),
        edge_conditions=_mapping(spec.pop("edge_conditions", None)),
        active_conditions=_mapping(spec.pop("active_conditions", None)),
        targets=arrays.pop("targets", None),
        target_kind=spec.pop("target_kind", None),
        provenance=spec.pop("provenance", {}),
        capacities=capacities,
    )
    if args.command == "prepare-ranked":
        bundle = prepare_ranked(
            **common,
            cell_ids=arrays.pop("cell_ids"), gene_ids=arrays.pop("gene_ids"),
            rank_positions=arrays.pop("rank_positions"),
            token_ids=arrays.pop("token_ids", None),
        )
    else:
        bundle = prepare_events(
            **common,
            cell_ids=arrays.pop("cell_ids"), entity_ids=arrays.pop("entity_ids"),
            occurrence_ids=arrays.pop("occurrence_ids"),
            event_ordinals=arrays.pop("event_ordinals"),
            timestamps=arrays.pop("timestamps", None),
            event_values=arrays.pop("event_values", None),
            position_windows=_mapping(spec.pop("position_windows", None)),
            timestamp_windows=_mapping(spec.pop("timestamp_windows", None)),
        )
    if spec:
        parser.error(f"unrecognized spec fields: {', '.join(sorted(spec))}")
    if arrays:
        parser.error(f"unconsumed arrays: {', '.join(sorted(arrays))}")
    manifest = write_raw_bundle(bundle, args.output)
    print(json.dumps({"manifest": str(manifest), "summary": bundle.summary},
                     sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
