#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = Path("/home/tumlinson/.agents/skills/public-omics-intake/scripts")
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from query_geo import resolve_geo_records  # type: ignore  # noqa: E402
from query_sra import resolve_sra_records  # type: ignore  # noqa: E402
from rank_candidates import rank_records  # type: ignore  # noqa: E402


OUT_ROOT = REPO_ROOT / "docs" / "public_omics"

BIOLOGY_FIRST_WEIGHTS: dict[str, float] = {
    "species_match": 15.0,
    "stage_match": 16.0,
    "modality_match": 12.0,
    "assay_compatibility": 10.0,
    "processed_matrices": 6.0,
    "raw_files": 5.0,
    "metadata_richness": 10.0,
    "linked_accessions": 7.0,
    "public_access": 10.0,
    "integration_ease": 6.0,
    "identifier_consistency": 3.0,
}


POOL_SPECS: dict[str, dict[str, Any]] = {
    "mouse_scrna_series": {
        "title": "Mouse embryo single-cell RNA series",
        "summary": (
            "Primary mouse RNA anchors intended to be stitched into one broad developmental "
            "path from fertilization through gastrulation and organogenesis."
        ),
        "query_spec": {
            "description": (
                "Mouse embryo scRNA-seq series spanning fertilization, preimplantation, "
                "gastrulation, and organogenesis for Cellerator developmental backbone testing."
            ),
            "biological_system": "Mouse embryogenesis",
            "organisms": ["Mus musculus"],
            "developmental_stages": [
                "zygote",
                "2-cell",
                "4-cell",
                "8-cell",
                "morula",
                "blastocyst",
                "implantation",
                "gastrulation",
                "organogenesis",
            ],
            "preferred_modalities": ["scrna-seq"],
            "required_modalities": [],
            "processed_files_acceptable": True,
            "raw_files_required": False,
            "public_only_required": True,
            "intended_use": "Mouse developmental backbone, real-data ingest, preprocessing, and runtime benchmarking",
        },
        "geo_accessions": [
            "GSE45719",
            "GSE87038",
            "GSE119945",
            "GSE186068",
            "GSE186069",
            "GSE278603",
        ],
        "sra_accessions": [],
        "curation": {
            "GSE45719": {
                "stage": "zygote 2-cell 4-cell 8-cell morula blastocyst preimplantation",
                "modality": ["scrna-seq"],
                "assay": ["single-cell RNA-seq"],
                "recommended_tier": "primary",
                "coverage_role": "preimplantation anchor",
                "notes": [
                    "Extends the mouse ladder back to fertilization and early cleavage stages.",
                    "Useful for testing low-cell-count early embryo handling before atlas-scale ingestion.",
                ],
            },
            "GSE87038": {
                "stage": "gastrulation early organogenesis E6.5 E7.0 E7.5 E8.0 E8.5",
                "modality": ["scrna-seq"],
                "assay": ["single-cell RNA-seq"],
                "recommended_tier": "primary",
                "coverage_role": "gastrulation anchor",
                "notes": [
                    "Strong bridge between early embryo states and lineage diversification.",
                    "Canonical gastrulation-scale reference for backbone structure validation.",
                ],
            },
            "GSE119945": {
                "stage": "organogenesis E9.5 E10.5 E11.5 E12.5 E13.5",
                "modality": ["scrna-seq"],
                "assay": ["single-cell RNA-seq"],
                "recommended_tier": "primary",
                "coverage_role": "organogenesis anchor",
                "notes": [
                    "Large downstream mouse atlas that extends the developmental path into organogenesis.",
                    "Good late-stage runtime and storage stress test candidate.",
                ],
            },
            "GSE186068": {
                "stage": "organogenesis fetal mouse embryo",
                "modality": ["scrna-seq"],
                "assay": ["single-cell RNA-seq"],
                "recommended_tier": "secondary",
                "coverage_role": "late-stage alternative",
                "notes": [
                    "Alternative late-stage RNA atlas for robustness and benchmark replication.",
                ],
            },
            "GSE186069": {
                "stage": "organogenesis fetal mouse embryo",
                "modality": ["scrna-seq"],
                "assay": ["single-cell RNA-seq"],
                "recommended_tier": "secondary",
                "coverage_role": "late-stage alternative",
                "notes": [
                    "Paired alternative to GSE186068 for redundancy in downstream organogenesis coverage.",
                ],
            },
            "GSE278603": {
                "stage": "gastrulation organogenesis early mouse embryo spatial",
                "modality": ["scrna-seq", "spatial-transcriptomics"],
                "assay": ["single-cell RNA-seq", "spatial transcriptomics"],
                "recommended_tier": "secondary",
                "coverage_role": "spatial bridge",
                "notes": [
                    "Spatially resolved adjunct for later figure-building and neighborhood validation.",
                ],
            },
        },
    },
    "mouse_secondary_modalities": {
        "title": "Mouse embryo multiome and secondary modalities",
        "summary": (
            "Mouse datasets that add chromatin, methylation, or spatial structure around the "
            "core scRNA developmental path."
        ),
        "query_spec": {
            "description": (
                "Mouse embryo multiome and secondary-modality datasets for frozen-latent "
                "prediction, multimodal benchmarking, and figure support."
            ),
            "biological_system": "Mouse embryogenesis",
            "organisms": ["Mus musculus"],
            "developmental_stages": ["gastrulation", "organogenesis", "implantation"],
            "preferred_modalities": [
                "multiome",
                "scrna-seq",
                "scatac-seq",
                "spatial-transcriptomics",
                "scdna-methylome",
            ],
            "required_modalities": [],
            "processed_files_acceptable": True,
            "raw_files_required": False,
            "public_only_required": True,
            "intended_use": "Secondary-observation prediction, multimodal benchmarking, and figure construction",
        },
        "geo_accessions": [
            "GSE121708",
            "GSE216371",
            "GSE278603",
        ],
        "sra_accessions": [],
        "curation": {
            "GSE121708": {
                "stage": "gastrulation E4.5 E5.5 E6.5 E7.5",
                "modality": ["multiome", "scrna-seq", "scatac-seq", "scdna-methylome"],
                "assay": ["scNMT-seq", "single-cell RNA-seq", "chromatin accessibility", "DNA methylation"],
                "recommended_tier": "primary",
                "coverage_role": "true multiome anchor",
                "notes": [
                    "Best first mouse multimodal anchor because the study is explicitly multi-omic at single-cell resolution.",
                    "Direct fit for the manuscript's frozen-latent secondary-observation story.",
                ],
            },
            "GSE216371": {
                "stage": "organogenesis chromatin accessibility mouse embryo",
                "modality": ["scatac-seq", "scrna-seq"],
                "assay": ["single-cell ATAC-seq"],
                "recommended_tier": "primary",
                "coverage_role": "chromatin atlas",
                "notes": [
                    "Good chromatin-focused complement to the RNA anchors for modality transfer tests.",
                ],
            },
            "GSE278603": {
                "stage": "gastrulation organogenesis early mouse embryo spatial",
                "modality": ["scrna-seq", "spatial-transcriptomics"],
                "assay": ["single-cell RNA-seq", "spatial transcriptomics"],
                "recommended_tier": "secondary",
                "coverage_role": "spatial adjunct",
                "notes": [
                    "Spatial modality is useful for manuscript figures even if it is not a canonical multiome pair.",
                ],
            },
        },
    },
    "primate_scrna_series": {
        "title": "Human and non-human primate embryo single-cell RNA series",
        "summary": (
            "Primate RNA references for alignment, projection, and cross-species comparison, "
            "with human first and macaque follow-on coverage."
        ),
        "query_spec": {
            "description": (
                "Human and non-human primate embryo scRNA-seq datasets spanning "
                "preimplantation through gastrulation or organogenesis."
            ),
            "biological_system": "Primate embryogenesis",
            "organisms": ["Homo sapiens", "Macaca fascicularis", "Macaca mulatta"],
            "developmental_stages": [
                "zygote",
                "morula",
                "blastocyst",
                "implantation",
                "gastrulation",
                "organogenesis",
            ],
            "preferred_modalities": ["scrna-seq", "spatial-transcriptomics"],
            "required_modalities": [],
            "processed_files_acceptable": True,
            "raw_files_required": False,
            "public_only_required": True,
            "intended_use": "Primate alignment, projection figures, and cross-species evaluation",
        },
        "geo_accessions": [
            "GSE36552",
            "GSE109555",
            "GSE136447",
            "GSE157329",
            "GSE103313",
            "GSE117219",
            "GSE193007",
        ],
        "sra_accessions": [],
        "curation": {
            "GSE36552": {
                "stage": "zygote 2-cell 4-cell 8-cell morula blastocyst preimplantation",
                "modality": ["scrna-seq"],
                "assay": ["single-cell RNA-seq"],
                "recommended_tier": "secondary",
                "coverage_role": "human preimplantation anchor",
                "notes": [
                    "Classical early human embryo RNA anchor that reaches back to fertilization-stage material.",
                ],
            },
            "GSE109555": {
                "stage": "implantation peri-implantation DNA methylation transcriptome",
                "modality": ["scrna-seq", "scdna-methylome"],
                "assay": ["single-cell RNA-seq", "DNA methylation"],
                "recommended_tier": "primary",
                "coverage_role": "human implantation bridge",
                "notes": [
                    "Valuable because it links transcriptome and methylome structure around implantation.",
                ],
            },
            "GSE136447": {
                "stage": "pre-gastrulation post-implantation day 8 day 10 day 12 day 14",
                "modality": ["scrna-seq", "spatial-transcriptomics"],
                "assay": ["single-cell RNA-seq", "spatial transcriptomics"],
                "recommended_tier": "primary",
                "coverage_role": "human post-implantation anchor",
                "notes": [
                    "Strong human bridge from peri-implantation into pre-gastrulation states.",
                ],
            },
            "GSE157329": {
                "stage": "human early organogenesis gastrulation CS12 CS13 CS14 CS15 CS16",
                "modality": ["scrna-seq", "spatial-transcriptomics"],
                "assay": ["single-cell RNA-seq", "spatial transcriptomics"],
                "recommended_tier": "primary",
                "coverage_role": "human organogenesis anchor",
                "notes": [
                    "Large human follow-on atlas for later-stage cross-species projection and figure work.",
                ],
            },
            "GSE103313": {
                "stage": "rhesus macaque oocyte preimplantation embryo",
                "modality": ["scrna-seq"],
                "assay": ["single-cell RNA-seq"],
                "recommended_tier": "secondary",
                "coverage_role": "macaque preimplantation anchor",
                "notes": [
                    "Useful non-human primate comparator for the earliest developmental states.",
                ],
            },
            "GSE117219": {
                "stage": "rhesus macaque preimplantation blastocyst embryo",
                "modality": ["scrna-seq"],
                "assay": ["single-cell RNA-seq"],
                "recommended_tier": "secondary",
                "coverage_role": "macaque preimplantation alternative",
                "notes": [
                    "Secondary non-human primate RNA anchor for preimplantation validation.",
                ],
            },
            "GSE193007": {
                "stage": "cynomolgus gastrulation early organogenesis E16 E17 E18 E20",
                "modality": ["scrna-seq"],
                "assay": ["single-cell RNA-seq"],
                "recommended_tier": "primary",
                "coverage_role": "macaque gastrulation anchor",
                "notes": [
                    "Best non-human primate in vivo-like gastrulation/early organogenesis anchor in this first pass.",
                ],
            },
        },
    },
    "primate_secondary_modalities": {
        "title": "Human and non-human primate secondary modalities",
        "summary": (
            "Primate methylome, chromatin, or spatial datasets that can support multimodal "
            "prediction tests and figure building where true embryo multiome data are scarce."
        ),
        "query_spec": {
            "description": (
                "Primate embryo secondary-modality datasets including methylome, chromatin "
                "accessibility, and spatial data."
            ),
            "biological_system": "Primate embryogenesis",
            "organisms": ["Homo sapiens", "Macaca fascicularis", "Macaca mulatta"],
            "developmental_stages": ["preimplantation", "implantation", "gastrulation", "organogenesis"],
            "preferred_modalities": [
                "multiome",
                "scatac-seq",
                "scdna-methylome",
                "spatial-transcriptomics",
                "scrna-seq",
            ],
            "required_modalities": [],
            "processed_files_acceptable": True,
            "raw_files_required": False,
            "public_only_required": True,
            "intended_use": "Secondary-observation prediction and cross-species multimodal support",
        },
        "geo_accessions": [
            "GSE49828",
            "GSE60166",
            "GSE81233",
            "GSE182838",
        ],
        "sra_accessions": [
            "PRJNA512422",
        ],
        "curation": {
            "GSE49828": {
                "stage": "human preimplantation blastocyst DNA methylation",
                "modality": ["scdna-methylome"],
                "assay": ["reduced representation bisulfite sequencing"],
                "recommended_tier": "secondary",
                "coverage_role": "human methylome anchor",
                "notes": [
                    "Human preimplantation methylome support set; useful even though it is not a same-cell multiome assay.",
                ],
            },
            "GSE60166": {
                "stage": "monkey preimplantation blastocyst DNA methylation",
                "modality": ["scdna-methylome"],
                "assay": ["DNA methylation sequencing"],
                "recommended_tier": "secondary",
                "coverage_role": "macaque methylome anchor",
                "notes": [
                    "Non-human primate methylome companion for early developmental comparisons.",
                ],
            },
            "GSE81233": {
                "stage": "human preimplantation DNA methylome",
                "modality": ["scdna-methylome"],
                "assay": ["single-cell DNA methylome sequencing"],
                "recommended_tier": "secondary",
                "coverage_role": "human methylome alternative",
                "notes": [
                    "Alternate human methylome dataset with explicit single-cell methylome framing.",
                ],
            },
            "GSE182838": {
                "stage": "cynomolgus gastrulating embryo spatial",
                "modality": ["spatial-transcriptomics"],
                "assay": ["spatial transcriptomics"],
                "recommended_tier": "primary",
                "coverage_role": "macaque spatial anchor",
                "notes": [
                    "Best primate spatial support set found in this first pass.",
                ],
            },
            "PRJNA512422": {
                "stage": "cynomolgus post-implantation gastrulation E13 E16 E20 chromatin accessibility",
                "modality": ["multiome", "scatac-seq", "scrna-seq"],
                "assay": ["ATAC-seq", "single-cell omics"],
                "title": "Program of monkey embryogenesis and germ cell formation in vitro",
                "recommended_tier": "primary",
                "coverage_role": "macaque chromatin anchor",
                "notes": [
                    "Official SRA study abstract explicitly describes both scATAC-seq and single-cell omics in cynomolgus embryos.",
                    "Use this as the leading primate chromatin-support candidate while direct human embryo multiome remains sparse.",
                ],
            },
        },
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rendered = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, bool):
                    rendered[key] = "true" if value else "false"
                elif isinstance(value, (list, dict)):
                    rendered[key] = json.dumps(value, sort_keys=True)
                elif value is None:
                    rendered[key] = ""
                else:
                    rendered[key] = str(value)
            writer.writerow(rendered)


def merge_record(base: dict[str, Any], override: dict[str, Any], pool_id: str) -> dict[str, Any]:
    record = dict(base)
    record["pool_id"] = pool_id
    record["pool_title"] = POOL_SPECS[pool_id]["title"]
    for key, value in override.items():
        if key == "notes":
            record[key] = list(value)
        else:
            record[key] = value
    return record


def flatten_summary(record: dict[str, Any]) -> dict[str, Any]:
    breakdown = record.get("ranking_breakdown", {})
    return {
        "primary_accession": record.get("primary_accession", ""),
        "pool_id": record.get("pool_id", ""),
        "recommended_tier": record.get("recommended_tier", ""),
        "coverage_role": record.get("coverage_role", ""),
        "title": record.get("title", ""),
        "species": "|".join(record.get("species", [])),
        "modality": "|".join(record.get("modality", [])),
        "stage": record.get("stage", ""),
        "integratability_score": record.get("integratability_score", 0.0),
        "processed_available": record.get("processed_available", False),
        "raw_available": record.get("raw_available", False),
        "stage_match_score": breakdown.get("stage_match", {}).get("score", ""),
        "modality_match_score": breakdown.get("modality_match", {}).get("score", ""),
        "species_match_score": breakdown.get("species_match", {}).get("score", ""),
        "ranking_rationale": " | ".join(record.get("ranking_rationale", [])),
        "notes": " | ".join(record.get("notes", [])),
    }


def build_geo_catalog(all_accessions: list[str]) -> dict[str, dict[str, Any]]:
    _, records = resolve_geo_records(
        description=None,
        accessions=all_accessions,
        retmax=max(50, len(all_accessions) * 4),
    )
    wanted = {accession.upper() for accession in all_accessions}
    catalog: dict[str, dict[str, Any]] = {}
    for record in records:
        accession = record.get("primary_accession", "").upper()
        if accession.startswith("GSE") and accession in wanted:
            catalog[accession] = record
    return catalog


def build_sra_catalog(all_accessions: list[str]) -> dict[str, dict[str, Any]]:
    _, records, _ = resolve_sra_records(
        description=None,
        accessions=all_accessions,
        retmax=max(20, len(all_accessions) * 4),
    )
    wanted = {accession.upper() for accession in all_accessions}
    catalog: dict[str, dict[str, Any]] = {}
    for record in records:
        accession = record.get("primary_accession", "").upper()
        if accession in wanted:
            catalog[accession] = record
    return catalog


def main() -> int:
    geo_accessions = sorted(
        {
            accession
            for spec in POOL_SPECS.values()
            for accession in spec.get("geo_accessions", [])
        }
    )
    sra_accessions = sorted(
        {
            accession
            for spec in POOL_SPECS.values()
            for accession in spec.get("sra_accessions", [])
        }
    )

    geo_catalog = build_geo_catalog(geo_accessions)
    sra_catalog = build_sra_catalog(sra_accessions)

    write_json(OUT_ROOT / "query_specs" / "biology_first_weights.json", BIOLOGY_FIRST_WEIGHTS)
    write_json(
        OUT_ROOT / "query_specs" / "candidate_catalog.json",
        {
            "geo_accessions": geo_accessions,
            "sra_accessions": sra_accessions,
        },
    )

    combined_rows: list[dict[str, Any]] = []
    pool_summary: dict[str, Any] = {}

    for pool_id, spec in POOL_SPECS.items():
        curated_records: list[dict[str, Any]] = []
        for accession in spec.get("geo_accessions", []):
            base = geo_catalog[accession]
            curated_records.append(merge_record(base, spec["curation"][accession], pool_id))
        for accession in spec.get("sra_accessions", []):
            base = sra_catalog[accession]
            curated_records.append(merge_record(base, spec["curation"][accession], pool_id))

        query_spec = dict(spec["query_spec"])
        query_spec["pool_id"] = pool_id
        query_spec["title"] = spec["title"]
        query_spec["summary"] = spec["summary"]
        ranked = rank_records(curated_records, query_spec, BIOLOGY_FIRST_WEIGHTS)

        payload = {
            "pool_id": pool_id,
            "title": spec["title"],
            "summary": spec["summary"],
            "query_spec": query_spec,
            "weights": BIOLOGY_FIRST_WEIGHTS,
            "candidates": ranked,
        }
        write_json(OUT_ROOT / "query_specs" / f"{pool_id}.json", query_spec)
        write_json(OUT_ROOT / "ranked" / f"{pool_id}.json", payload)
        write_tsv(OUT_ROOT / "ranked" / f"{pool_id}.tsv", [flatten_summary(row) for row in ranked])

        combined_rows.extend(flatten_summary(row) for row in ranked)
        pool_summary[pool_id] = {
            "title": spec["title"],
            "summary": spec["summary"],
            "top_accessions": [row["primary_accession"] for row in ranked[:3]],
        }

    write_tsv(OUT_ROOT / "ranked" / "all_candidates.tsv", combined_rows)
    write_json(OUT_ROOT / "ranked" / "pool_summary.json", pool_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
