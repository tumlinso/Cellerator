from ._cellerator import (
    browse_cache_summary,
    builder_path_role,
    builder_path_role_name,
    codec_summary,
    conversion_report,
    dataset_partition_summary,
    dataset_shard_summary,
    dataset_summary,
    draft_dataset,
    embedded_metadata_dataset_summary,
    embedded_metadata_table,
    execution_format,
    execution_format_name,
    export_manifest,
    filesystem_entry,
    format_name,
    infer_builder_path_role,
    ingest_plan,
    ingest_policy,
    inspect_directory,
    inspect_manifest,
    inspect_sources,
    issue,
    issue_severity,
    list_filesystem_entries,
    load_embedded_metadata_table,
    load_observation_metadata_table,
    load_persisted_preprocess_table,
    manifest_inspection,
    observation_metadata_column,
    observation_metadata_column_summary,
    observation_metadata_summary,
    observation_metadata_table,
    persisted_preprocess_summary,
    persisted_preprocess_table,
    planned_dataset,
    planned_part,
    planned_shard,
    plan_ingest,
    preprocess,
    preprocess_config,
    preprocess_summary,
    run_event,
    runtime_service_summary,
    severity_name,
    source_dataset_summary,
    source_entry,
    sources_from_drafts,
    summarize_dataset,
    discover_drafts,
    convert,
)

__version__ = "0.1.0"


def _issues_message(issues):
    if not issues:
        return "operation failed"
    return "; ".join(f"{severity_name(entry.severity)}:{entry.scope}:{entry.message}" for entry in issues)


def open(path: str):
    import cellshard

    return cellshard.open(path)


class DatasetBuilder:
    """Small high-level wrapper around the dataset workbench planning surface."""

    def __init__(self, inspection: manifest_inspection):
        self.inspection = inspection
        self.plan_result = None

    @classmethod
    def from_manifest(cls, manifest_path: str, reader_bytes: int = 8 << 20):
        return cls(inspect_manifest(manifest_path, reader_bytes=reader_bytes))

    @classmethod
    def from_directory(cls, dir_path: str, reader_bytes: int = 8 << 20):
        return cls(inspect_directory(dir_path, reader_bytes=reader_bytes))

    @classmethod
    def from_sources(cls, sources, label: str = "<builder>", reader_bytes: int = 8 << 20):
        return cls(inspect_sources(sources, label=label, reader_bytes=reader_bytes))

    @property
    def sources(self):
        return self.inspection.sources

    def plan(self, policy: ingest_policy | None = None):
        chosen_policy = policy if policy is not None else ingest_policy()
        self.plan_result = plan_ingest(self.sources, chosen_policy)
        return self.plan_result

    def convert(self, plan_result: ingest_plan | None = None):
        active_plan = plan_result if plan_result is not None else self.plan_result
        if active_plan is None:
            active_plan = self.plan()
        if not active_plan.ok:
            raise RuntimeError(_issues_message(active_plan.issues))
        report = convert(active_plan)
        if not report.ok:
            raise RuntimeError(_issues_message(report.issues))
        return report

    def inspect_output(self, path: str | None = None):
        dataset_path = path
        if dataset_path is None:
            if self.plan_result is None:
                raise ValueError("no ingest plan is available")
            dataset_path = self.plan_result.policy.output_path
        return summarize_dataset(dataset_path)

    def preprocess(self, path: str | None = None, config: preprocess_config | None = None):
        dataset_path = path
        if dataset_path is None:
            if self.plan_result is None:
                raise ValueError("no ingest plan is available")
            dataset_path = self.plan_result.policy.output_path
        chosen_config = config if config is not None else preprocess_config()
        result = preprocess(dataset_path, chosen_config)
        if not result.ok:
            raise RuntimeError(_issues_message(result.issues))
        return result

    def open(self, path: str | None = None):
        dataset_path = path
        if dataset_path is None:
            if self.plan_result is None:
                raise ValueError("no ingest plan is available")
            dataset_path = self.plan_result.policy.output_path
        return open(dataset_path)


__all__ = [
    "DatasetBuilder",
    "browse_cache_summary",
    "builder_path_role",
    "builder_path_role_name",
    "codec_summary",
    "conversion_report",
    "convert",
    "dataset_partition_summary",
    "dataset_shard_summary",
    "dataset_summary",
    "discover_drafts",
    "draft_dataset",
    "embedded_metadata_dataset_summary",
    "embedded_metadata_table",
    "execution_format",
    "execution_format_name",
    "export_manifest",
    "filesystem_entry",
    "format_name",
    "infer_builder_path_role",
    "ingest_plan",
    "ingest_policy",
    "inspect_directory",
    "inspect_manifest",
    "inspect_sources",
    "issue",
    "issue_severity",
    "list_filesystem_entries",
    "load_embedded_metadata_table",
    "load_observation_metadata_table",
    "load_persisted_preprocess_table",
    "manifest_inspection",
    "observation_metadata_column",
    "observation_metadata_column_summary",
    "observation_metadata_summary",
    "observation_metadata_table",
    "open",
    "persisted_preprocess_summary",
    "persisted_preprocess_table",
    "plan_ingest",
    "planned_dataset",
    "planned_part",
    "planned_shard",
    "preprocess",
    "preprocess_config",
    "preprocess_summary",
    "run_event",
    "runtime_service_summary",
    "severity_name",
    "source_dataset_summary",
    "source_entry",
    "sources_from_drafts",
    "summarize_dataset",
    "__version__",
]

